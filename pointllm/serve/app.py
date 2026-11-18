import asyncio
import hmac
import json
import logging
import queue
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from .backend import PointLLMBackend, check_cancel
from .config import Settings
from .engine import Engine, wait_for_job
from .errors import APIError
from .middleware import ChatBodyLimit
from .point_clouds import CloudStore, preprocess_cloud
from .schemas import ChatRequest

logger = logging.getLogger(__name__)


def public_error(exc):
    if isinstance(exc, APIError):
        return exc
    return APIError("PointLLM inference failed; see server logs", 500, "inference_error")


class GenerationStreamResponse(StreamingResponse):
    def __init__(self, job, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job = job

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # A disconnect while sending headers can precede the generator's
            # first iteration, so its own finally block would never execute.
            self.job.cancel.set()


def create_app(settings=None, backend=None):
    settings = settings or Settings.from_env()
    backend = backend if backend is not None else PointLLMBackend(settings)
    store = CloudStore(settings)
    upload_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.ready = False
        try:
            await run_in_threadpool(backend.load)
            app.state.engine = Engine(backend, settings)
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            if hasattr(app.state, "engine"):
                await app.state.engine.close()
            else:
                backend.close()

    app = FastAPI(title="PointLLM", lifespan=lifespan)
    app.add_middleware(
        ChatBodyLimit, max_bytes=settings.max_request_bytes,
        timeout=settings.request_timeout_seconds,
    )
    app.state.ready = False
    app.state.cloud_store = store

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        details = "; ".join(
            ".".join(str(part) for part in item["loc"]) + ": " + item["msg"]
            for item in exc.errors()
        )
        error = APIError(details, param="request")
        return JSONResponse(error.body(), status_code=400)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        error = APIError(str(exc.detail), exc.status_code, "http_error")
        return JSONResponse(error.body(), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        logger.exception("Unhandled serving error")
        error = public_error(exc)
        return JSONResponse(error.body(), status_code=error.status)

    async def authorize(request: Request):
        if settings.api_key:
            actual = request.headers.get("authorization", "").encode()
            expected = ("Bearer " + settings.api_key).encode()
            if not hmac.compare_digest(actual, expected):
                raise APIError("Invalid API key", 401, "invalid_api_key")
        if not app.state.ready:
            raise APIError("Model is not ready", 503, "not_ready")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        if not app.state.ready:
            raise APIError("Model is not ready", 503, "not_ready")
        return {"status": "ready"}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return {
            "object": "list",
            "data": [{"id": settings.model_alias, "object": "model", "created": 0, "owned_by": "pointllm"}],
        }

    @app.post("/v1/point-clouds", dependencies=[Depends(authorize)])
    async def upload(request: Request):
        if upload_lock.locked():
            raise APIError("Another cloud is being processed; retry later", 429, "server_busy")
        async with upload_lock:
            content_type = request.headers.get("content-type", "").split(";")[0].lower()
            if content_type not in ("application/octet-stream", "application/x-npy"):
                raise APIError("Send raw NPY bytes with Content-Type: application/octet-stream", 415)
            async def read_upload():
                payload = bytearray()
                async for chunk in request.stream():
                    if len(payload) + len(chunk) > settings.max_upload_bytes:
                        raise APIError("Point cloud exceeds upload byte limit", 413, "upload_too_large")
                    payload.extend(chunk)
                return payload

            try:
                payload = await asyncio.wait_for(read_upload(), settings.request_timeout_seconds)
            except asyncio.TimeoutError as exc:
                raise APIError("Upload deadline exceeded", 408, "request_timeout") from exc
            points = await run_in_threadpool(preprocess_cloud, bytes(payload), settings)
            return store.add(points).metadata()

    @app.delete("/v1/point-clouds/{cloud_id}", dependencies=[Depends(authorize)])
    async def delete(cloud_id: str):
        store.delete(cloud_id)
        return {"id": cloud_id, "object": "point_cloud", "deleted": True}

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat(body: ChatRequest, request: Request):
        if body.model != settings.model_alias:
            raise APIError("Unknown model", 404, "model_not_found", "model")
        cloud = store.get(body.point_cloud_id)
        job = app.state.engine.submit(body, cloud.points)
        try:
            await wait_for_job(job, job.ready, request)
        except Exception as exc:
            raise public_error(exc) from exc
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        def envelope(stream=False):
            return {
                "id": completion_id,
                "object": "chat.completion.chunk" if stream else "chat.completion",
                "created": created,
                "model": settings.model_alias,
            }

        if not body.stream:
            try:
                result = await wait_for_job(job, job.future, request)
            except Exception as exc:
                raise public_error(exc) from exc
            return {
                **envelope(),
                "choices": [{
                    "index": 0, "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason,
                }],
                "usage": result.usage(),
            }

        def event(choices, usage=None):
            data = {**envelope(True), "choices": choices}
            if body.stream_options and body.stream_options.include_usage:
                data["usage"] = usage
            return "data: " + json.dumps(data) + "\n\n"

        async def events():
            try:
                yield event([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
                while True:
                    check_cancel(job.cancel, job.deadline)
                    if await request.is_disconnected():
                        return
                    try:
                        chunk = job.chunks.get_nowait()
                    except queue.Empty:
                        if job.future.done():
                            result = job.future.result()
                            yield event([{"index": 0, "delta": {}, "finish_reason": result.finish_reason}])
                            if body.stream_options and body.stream_options.include_usage:
                                yield event([], result.usage())
                            break
                        await asyncio.sleep(0.02)
                    else:
                        yield event([{"index": 0, "delta": {"content": chunk}, "finish_reason": None}])
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield "data: " + json.dumps(public_error(exc).body()) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                job.cancel.set()

        return GenerationStreamResponse(
            job, events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app

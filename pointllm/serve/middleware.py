import asyncio
import time

from fastapi.responses import JSONResponse

from .errors import APIError


class ChatBodyLimit:
    """Bound JSON buffering before FastAPI parses the body (also for chunked HTTP)."""

    def __init__(self, app, max_bytes, timeout):
        self.app = app
        self.max_bytes = max_bytes
        self.timeout = timeout

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].rstrip("/") != "/v1/chat/completions":
            return await self.app(scope, receive, send)
        body = bytearray()
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                message = await asyncio.wait_for(receive(), remaining)
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > self.max_bytes:
                    raise APIError("Chat body exceeds byte limit", 413, "request_too_large")
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
        except asyncio.TimeoutError:
            error = APIError("Request body deadline exceeded", 408, "request_timeout")
            return await JSONResponse(error.body(), status_code=error.status)(scope, receive, send)
        except APIError as error:
            return await JSONResponse(error.body(), status_code=error.status)(scope, receive, send)

        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

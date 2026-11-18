import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from starlette.requests import ClientDisconnect

from pointllm.serve.app import GenerationStreamResponse
from pointllm.serve.middleware import ChatBodyLimit


def test_body_read_timeout():
    async def exercise():
        sent = []

        async def downstream(scope, receive, send):
            raise AssertionError("Timed-out requests must not reach FastAPI")

        async def receive():
            await asyncio.sleep(0.1)
            return {"type": "http.request", "body": b"{}", "more_body": False}

        async def send(message):
            sent.append(message)

        middleware = ChatBodyLimit(downstream, max_bytes=64, timeout=0.01)
        await middleware(
            {"type": "http", "path": "/v1/chat/completions", "asgi": {"spec_version": "2.4"}},
            receive, send,
        )
        assert sent[0]["status"] == 408
        assert json.loads(sent[1]["body"])["error"]["code"] == "request_timeout"

    asyncio.run(exercise())


def test_disconnect_before_stream_iteration_cancels_job():
    async def exercise():
        job = SimpleNamespace(cancel=threading.Event())
        response = GenerationStreamResponse(job, iter(["text"]))

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            raise OSError("Client disconnected before headers")

        with pytest.raises((OSError, ClientDisconnect)):
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}},
                receive, send,
            )
        assert job.cancel.is_set()

    asyncio.run(exercise())

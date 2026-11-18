import asyncio
import threading

import pytest

from pointllm.serve.config import Settings
from pointllm.serve.engine import Engine, wait_for_job
from pointllm.serve.errors import APIError
from pointllm.serve.schemas import ChatRequest
from conftest import chat_body


def test_admission_and_cancel_release_only_after_worker_finishes(backend, cloud_array):
    started = threading.Event()
    release = threading.Event()
    original = backend.generate

    def blocking(*args):
        started.set()
        release.wait(timeout=5)
        return original(*args)

    backend.generate = blocking

    async def exercise():
        engine = Engine(backend, Settings(max_pending_requests=1))
        request = ChatRequest(**chat_body("test"))
        job = engine.submit(request, cloud_array)
        while not started.is_set():
            await asyncio.sleep(0.005)
        job.cancel.set()
        with pytest.raises(APIError, match="full"):
            engine.submit(request, cloud_array)
        release.set()
        while not job.future.done():
            await asyncio.sleep(0.005)
        await asyncio.sleep(0)
        assert not engine.jobs
        await engine.close()

    asyncio.run(exercise())


def test_disconnect_signals_cancellation(backend, cloud_array):
    backend.delay = 0.2

    class Disconnected:
        async def is_disconnected(self):
            return True

    async def exercise():
        engine = Engine(backend, Settings())
        job = engine.submit(ChatRequest(**chat_body("test")), cloud_array)
        with pytest.raises(APIError, match="disconnected"):
            await wait_for_job(job, job.future, Disconnected())
        assert job.cancel.is_set()
        await engine.close()

    asyncio.run(exercise())


def test_stream_queue_backpressure_cancels(backend, cloud_array):
    entered = threading.Event()

    def flood(prepared, cancel, deadline, emit):
        entered.set()
        for index in range(100):
            emit(str(index))

    backend.generate = flood

    async def exercise():
        engine = Engine(backend, Settings())
        job = engine.submit(ChatRequest(**chat_body("test", stream=True)), cloud_array)
        while not entered.is_set() or job.chunks.qsize() < 32:
            await asyncio.sleep(0.005)
        assert not job.future.done()
        job.cancel.set()
        await engine.close()
        assert job.future.done()

    asyncio.run(exercise())

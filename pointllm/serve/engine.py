import asyncio
import concurrent.futures
import logging
import queue
import threading
import time

from .backend import check_cancel
from .errors import APIError

logger = logging.getLogger(__name__)


class Job:
    def __init__(self, timeout):
        self.cancel = threading.Event()
        self.deadline = time.monotonic() + timeout
        self.ready = concurrent.futures.Future()
        self.chunks = queue.Queue(maxsize=32)
        self.future = None

    def emit(self, text):
        while True:
            check_cancel(self.cancel, self.deadline)
            try:
                self.chunks.put(text, timeout=0.05)
                return
            except queue.Full:
                continue


class Engine:
    """One model thread; queued + running jobs are bounded by admission control."""

    def __init__(self, backend, settings):
        self.backend = backend
        self.settings = settings
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pointllm")
        self.jobs = set()
        self.closed = False

    def submit(self, request, cloud):
        if self.closed:
            raise APIError("Server is shutting down", 503, "not_ready")
        if len(self.jobs) >= self.settings.max_pending_requests:
            raise APIError("Generation queue is full", 429, "server_busy")
        job = Job(self.settings.request_timeout_seconds)
        loop = asyncio.get_running_loop()
        self.jobs.add(job)

        def run():
            try:
                check_cancel(job.cancel, job.deadline)
                prepared = self.backend.prepare(request, cloud)
                check_cancel(job.cancel, job.deadline)
                job.ready.set_result(prepared.prompt_tokens)
                return self.backend.generate(
                    prepared, job.cancel, job.deadline,
                    job.emit if request.stream else lambda text: None,
                )
            except Exception as exc:
                if not job.ready.done():
                    job.ready.set_exception(exc)
                if not isinstance(exc, APIError):
                    logger.exception("PointLLM generation failed")
                raise

        job.future = self.executor.submit(run)
        job.future.add_done_callback(lambda future: loop.call_soon_threadsafe(self.jobs.discard, job))
        return job

    async def close(self):
        self.closed = True
        for job in list(self.jobs):
            job.cancel.set()
        # Do not unload weights while their generation thread is still running.
        await asyncio.get_running_loop().run_in_executor(None, self.executor.shutdown, True)
        self.backend.close()


async def wait_for_job(job, future, request):
    try:
        while not future.done():
            check_cancel(job.cancel, job.deadline)
            if await request.is_disconnected():
                job.cancel.set()
                raise APIError("Client disconnected", 499, "request_cancelled")
            await asyncio.sleep(0.02)
        check_cancel(job.cancel, job.deadline)
        return future.result()
    except BaseException:
        job.cancel.set()
        raise

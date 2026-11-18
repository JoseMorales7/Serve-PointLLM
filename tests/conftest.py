import io
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pointllm.serve.app import create_app
from pointllm.serve.backend import GenerationResult, check_cancel
from pointllm.serve.config import Settings
from pointllm.serve.errors import APIError


def npy_bytes(points):
    output = io.BytesIO()
    np.save(output, points)
    return output.getvalue()


@pytest.fixture
def cloud_array():
    rng = np.random.default_rng(42)
    return np.concatenate(
        [rng.normal(size=(8192, 3)), rng.random((8192, 3))], axis=1
    ).astype(np.float32)


@pytest.fixture
def cloud_bytes(cloud_array):
    return npy_bytes(cloud_array)


class FakeBackend:
    def __init__(self):
        self.loads = 0
        self.closed = False
        self.requests = []
        self.delay = 0
        self.fail = False
        self.running = False
        self.cancelled = False

    def load(self):
        self.loads += 1

    def prepare(self, request, cloud):
        if request.output_limit > 1024:
            raise APIError("Context overflow", code="context_length_exceeded")
        self.requests.append((request, cloud))
        return SimpleNamespace(prompt_tokens=20)

    def generate(self, prepared, cancel, deadline, emit):
        self.running = True
        try:
            until = time.monotonic() + self.delay
            while time.monotonic() < until:
                check_cancel(cancel, deadline)
                time.sleep(0.005)
            if self.fail:
                raise RuntimeError("private backend detail")
            emit("A wooden ")
            emit("chair.")
            return GenerationResult("A wooden chair.", 20, 4, "stop")
        except APIError:
            self.cancelled = True
            raise
        finally:
            self.running = False

    def close(self):
        assert not self.running
        self.closed = True


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def app(backend):
    return create_app(Settings(), backend)


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


@pytest.fixture
def cloud_id(client, cloud_bytes):
    response = client.post(
        "/v1/point-clouds", content=cloud_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def chat_body(cloud_id, **kwargs):
    return {
        "model": "pointllm", "point_cloud_id": cloud_id,
        "messages": [{"role": "user", "content": "Describe this object."}],
        **kwargs,
    }


@pytest.fixture
def serve_http():
    """Run a real local HTTP server, allowing for slow checkpoint startup."""
    import uvicorn

    @contextmanager
    def serve(app, startup_timeout=10):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + startup_timeout
        try:
            while not server.started:
                if not thread.is_alive() or time.monotonic() >= deadline:
                    pytest.fail("Test HTTP server did not start")
                time.sleep(0.01)
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=30)
            sock.close()
            assert not thread.is_alive()

    return serve

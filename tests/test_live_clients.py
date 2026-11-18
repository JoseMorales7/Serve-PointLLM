import os
import subprocess
import threading

import httpx
import pytest

from pointllm.serve.app import create_app
from pointllm.serve.backend import GenerationResult
from pointllm.serve.config import Settings
from conftest import chat_body


@pytest.fixture
def live_server(backend, serve_http):
    app = create_app(Settings(), backend)
    with serve_http(app) as base_url:
        yield base_url


def upload(base_url, payload):
    response = httpx.post(
        base_url + "/v1/point-clouds", content=payload,
        headers={"Content-Type": "application/octet-stream"},
    )
    response.raise_for_status()
    return response.json()["id"]


def test_openai_sdk_over_http(live_server, cloud_bytes):
    openai = pytest.importorskip("openai")
    cloud_id = upload(live_server, cloud_bytes)
    with openai.OpenAI(base_url=live_server + "/v1", api_key="local") as client:
        assert client.models.list().data[0].id == "pointllm"
        kwargs = {
            "model": "pointllm",
            "messages": [{"role": "user", "content": "Describe this object."}],
            "extra_body": {"point_cloud_id": cloud_id},
            "max_completion_tokens": 16,
        }
        result = client.chat.completions.create(**kwargs)
        assert result.choices[0].message.content == "A wooden chair."
        with client.chat.completions.create(
            **kwargs, stream=True, stream_options={"include_usage": True}
        ) as stream:
            chunks = list(stream)
        assert "".join(
            choice.delta.content or "" for chunk in chunks for choice in chunk.choices
        ) == "A wooden chair."
        assert chunks[-1].usage.total_tokens == 24


def test_langchain_over_http(live_server, cloud_bytes):
    python = os.getenv("POINTLLM_CLIENT_PYTHON")
    if not python:
        pytest.skip("Set POINTLLM_CLIENT_PYTHON to a separate Python with langchain-openai installed")
    cloud_id = upload(live_server, cloud_bytes)
    subprocess.run([
        python, "-c", """
import sys
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="pointllm", base_url=sys.argv[1] + "/v1", api_key="local",
    temperature=0, max_tokens=16, use_responses_api=False,
)
cloud_llm = llm.bind(extra_body={"point_cloud_id": sys.argv[2]})
answer = cloud_llm.invoke("Describe this object.")
assert answer.content == "A wooden chair.", answer
followup = cloud_llm.invoke([
    ("human", "Describe this object."), answer, ("human", "What color is it?")
])
assert followup.content == "A wooden chair."
chunks = list(cloud_llm.stream("Describe this object."))
assert "".join(chunk.content for chunk in chunks) == "A wooden chair.", chunks
""", live_server, cloud_id,
    ], check=True, timeout=60)


def test_stream_is_incremental_over_http(live_server, cloud_bytes, backend):
    release = threading.Event()

    def generate(prepared, cancel, deadline, emit):
        emit("First ")
        assert release.wait(timeout=5), "HTTP client did not receive the first token in time"
        emit("second")
        return GenerationResult("First second", 20, 2, "stop")

    backend.generate = generate
    cloud_id = upload(live_server, cloud_bytes)
    try:
        with httpx.stream(
            "POST", live_server + "/v1/chat/completions",
            json=chat_body(cloud_id, stream=True), timeout=10,
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if "First " in line:
                    release.set()
                    break
            assert release.is_set()
    finally:
        release.set()


def test_stream_disconnect_cancels_worker(live_server, cloud_bytes, backend):
    cancelled = threading.Event()

    def generate(prepared, cancel, deadline, emit):
        emit("First ")
        if cancel.wait(timeout=5):
            cancelled.set()
        return GenerationResult("First ", 20, 1, "stop")

    backend.generate = generate
    cloud_id = upload(live_server, cloud_bytes)
    with httpx.stream(
        "POST", live_server + "/v1/chat/completions",
        json=chat_body(cloud_id, stream=True), timeout=10,
    ) as response:
        for line in response.iter_lines():
            if "First " in line:
                break
    assert cancelled.wait(timeout=5)

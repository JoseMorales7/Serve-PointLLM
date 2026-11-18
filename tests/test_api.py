import json
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from pointllm.serve.app import create_app
from pointllm.serve.config import Settings
from conftest import chat_body


def test_http_import_does_not_load_model_stack():
    subprocess.run(
        [sys.executable, "-c",
         "import sys; from pointllm.serve.app import create_app; "
         "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
         "assert 'pointllm.model' not in sys.modules"],
        check=True,
    )


def test_lifecycle_and_health(backend):
    app = create_app(Settings(), backend)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200
        assert client.get("/v1/models").json()["data"][0]["id"] == "pointllm"
        assert backend.loads == 1
    assert backend.closed
    assert not app.state.ready


def test_readiness_without_lifespan(app):
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/v1/models").json()["error"]["code"] == "not_ready"


def test_authentication(backend):
    app = create_app(Settings(api_key="secret"), backend)
    with TestClient(app) as client:
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            assert client.get("/v1/models", headers=headers).status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
        assert client.get("/health").status_code == 200


def test_upload_chat_followup_delete(client, cloud_id, backend):
    response = client.post("/v1/chat/completions", json=chat_body(cloud_id))
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "A wooden chair."}
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    history = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Describe it."},
        {"role": "assistant", "content": data["choices"][0]["message"]["content"]},
        {"role": "user", "content": "What color is it?"},
    ]
    assert client.post("/v1/chat/completions", json=chat_body(cloud_id, messages=history)).status_code == 200
    assert len(backend.requests[1][0].messages) == 4
    assert backend.requests[0][1] is backend.requests[1][1]
    assert backend.requests[0][1].shape == (8192, 6)
    assert not backend.requests[0][1].flags.writeable
    assert client.delete("/v1/point-clouds/" + cloud_id).json()["deleted"]
    assert client.post("/v1/chat/completions", json=chat_body(cloud_id)).status_code == 404
    assert client.delete("/v1/point-clouds/" + cloud_id).status_code == 404


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_output_limit_aliases(client, cloud_id, backend, field):
    assert client.post("/v1/chat/completions", json=chat_body(cloud_id, **{field: 64})).status_code == 200
    assert backend.requests[-1][0].output_limit == 64


@pytest.mark.parametrize("options", [
    {"tools": []}, {"response_format": {"type": "json_object"}},
    {"stop": "chair"}, {"n": 2}, {"temperature": -1},
    {"top_p": 0}, {"max_tokens": 0}, {"max_tokens": True},
    {"max_tokens": 10, "max_completion_tokens": 11},
    {"stream_options": {"include_usage": True}},
    {"messages": []},
    {"messages": [{"role": "assistant", "content": "hello"}]},
    {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]},
    {"messages": [{"role": "tool", "content": "hello"}]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]},
])
def test_reject_invalid_requests(client, cloud_id, options):
    response = client.post("/v1/chat/completions", json=chat_body(cloud_id, **options))
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_request_error"


def test_unknown_model_and_cloud(client):
    assert client.post("/v1/chat/completions", json=chat_body("bad", model="other")).status_code == 404
    assert client.post("/v1/chat/completions", json=chat_body("../../etc/passwd")).status_code == 404


def test_malformed_json_and_unknown_route(client):
    response = client.post("/v1/chat/completions", content=b"{", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert "error" in response.json()
    assert "error" in client.get("/not-a-route").json()


def test_chat_body_limit(backend):
    app = create_app(Settings(max_request_bytes=64), backend)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", content=iter([b"x" * 32, b"x" * 33]),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"


def sse_events(response):
    return [
        json.loads(line[6:]) if line != "data: [DONE]" else "[DONE]"
        for line in response.text.splitlines() if line.startswith("data: ")
    ]


def test_streaming(client, cloud_id):
    response = client.post("/v1/chat/completions", json=chat_body(
        cloud_id, stream=True, stream_options={"include_usage": True}
    ))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(event["object"] == "chat.completion.chunk" for event in chunks)
    assert len({event["id"] for event in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(
        choice["delta"].get("content", "") for event in chunks for choice in event["choices"]
    )
    assert text == "A wooden chair."
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 24


def test_context_error_precedes_stream_headers(client, cloud_id):
    for stream in (False, True):
        response = client.post("/v1/chat/completions", json=chat_body(cloud_id, max_tokens=1025, stream=stream))
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "context_length_exceeded"


def test_inference_failure_is_sanitized(client, cloud_id, backend):
    backend.fail = True
    response = client.post("/v1/chat/completions", json=chat_body(cloud_id))
    assert response.status_code == 500
    assert "private backend detail" not in response.text
    events = sse_events(client.post("/v1/chat/completions", json=chat_body(cloud_id, stream=True)))
    assert events[-1] == "[DONE]"
    assert events[-2]["error"]["code"] == "inference_error"


def test_request_timeout(backend, cloud_bytes):
    backend.delay = 0.2
    app = create_app(Settings(request_timeout_seconds=0.05), backend)
    with TestClient(app) as client:
        cloud_id = client.post(
            "/v1/point-clouds", content=cloud_bytes,
            headers={"Content-Type": "application/octet-stream"},
        ).json()["id"]
        response = client.post("/v1/chat/completions", json=chat_body(cloud_id))
        assert response.status_code == 504
    assert backend.cancelled
    assert backend.closed


def test_streaming_timeout_is_an_sse_error(backend, cloud_bytes):
    backend.delay = 0.2
    app = create_app(Settings(request_timeout_seconds=0.05), backend)
    with TestClient(app) as client:
        cloud_id = client.post(
            "/v1/point-clouds", content=cloud_bytes,
            headers={"Content-Type": "application/octet-stream"},
        ).json()["id"]
        response = client.post("/v1/chat/completions", json=chat_body(cloud_id, stream=True))
        assert response.status_code == 200
        events = sse_events(response)
        assert events[-2]["error"]["code"] == "request_timeout"
        assert events[-1] == "[DONE]"
    assert backend.cancelled


def test_startup_failure_cleans_up_backend(backend):
    def fail():
        raise RuntimeError("Cannot load checkpoint")

    backend.load = fail
    app = create_app(Settings(), backend)
    with pytest.raises(RuntimeError, match="Cannot load"):
        with TestClient(app):
            pass
    assert backend.closed
    assert not app.state.ready


def test_cloud_capacity_and_upload_limit(backend, cloud_bytes):
    app = create_app(Settings(max_clouds=1), backend)
    with TestClient(app) as client:
        headers = {"Content-Type": "application/octet-stream"}
        assert client.post("/v1/point-clouds", content=cloud_bytes, headers=headers).status_code == 200
        assert client.post("/v1/point-clouds", content=cloud_bytes, headers=headers).status_code == 429
        assert client.post("/v1/point-clouds", content=cloud_bytes).status_code == 415
    app = create_app(Settings(max_upload_bytes=64), backend)
    with TestClient(app) as client:
        assert client.post("/v1/point-clouds", content=cloud_bytes, headers=headers).status_code == 413

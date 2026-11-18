"""Opt-in real HTTP inference tests using locally cached pretrained weights."""

import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from pointllm.serve.app import create_app
from pointllm.serve.config import Settings
from conftest import chat_body, npy_bytes


def red_chair_cloud():
    """Sample box surfaces for a seat, back, and four legs (Y is up)."""
    rng = np.random.default_rng(42)
    boxes = [
        ((0, 0.55, 0), (1, 0.16, 1)),
        ((0, 1.1, -0.44), (1, 1, 0.12)),
        *[((x, 0.25, z), (0.12, 0.5, 0.12)) for x in (-0.4, 0.4) for z in (-0.4, 0.4)],
    ]
    areas = [2 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]) for _, size in boxes]
    parts = rng.choice(len(boxes), 8192, p=np.array(areas) / sum(areas))
    xyz = np.empty((8192, 3))
    for part, (center, dimensions) in enumerate(boxes):
        indices = np.flatnonzero(parts == part)
        dimensions = np.array(dimensions)
        face_weights = np.prod(dimensions) / dimensions
        faces = rng.choice(6, len(indices), p=np.repeat(face_weights / face_weights.sum(), 2) / 2)
        points = rng.uniform(-0.5, 0.5, (len(indices), 3))
        points[np.arange(len(indices)), faces // 2] = (faces % 2) - 0.5
        xyz[indices] = points * dimensions + center
    rgb = np.tile([0.85, 0.08, 0.05], (8192, 1))
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


def test_synthetic_chair_fixture():
    cloud = red_chair_cloud()
    assert cloud.shape == (8192, 6)
    assert cloud.dtype == np.float32
    assert np.isfinite(cloud).all()
    assert np.all(cloud[:, 3] > cloud[:, 4])
    assert np.ptp(cloud[:, 1]) > 1


@pytest.mark.gpu
def test_pretrained_endpoint_over_http(serve_http):
    checkpoint = os.getenv("POINTLLM_TEST_MODEL")
    if not checkpoint:
        pytest.skip("Set POINTLLM_TEST_MODEL to a local trained checkpoint")
    assert Path(checkpoint).is_dir()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    openai = pytest.importorskip("openai")
    settings = Settings(
        model_path=checkpoint,
        device=os.getenv("POINTLLM_TEST_DEVICE", "cuda:0"),
        dtype=os.getenv("POINTLLM_TEST_DTYPE", "float16"),
        request_timeout_seconds=180,
    )
    report = {
        "checkpoint": checkpoint,
        "dtype": settings.dtype,
        "input": "synthetic red chair: 8192 colored surface points; not a benchmark",
    }
    started = time.monotonic()
    with serve_http(create_app(settings), startup_timeout=600) as base_url:
        report["startup_seconds"] = round(time.monotonic() - started, 3)
        assert httpx.get(base_url + "/ready").status_code == 200
        response = httpx.post(
            base_url + "/v1/point-clouds", content=npy_bytes(red_chair_cloud()),
            headers={"Content-Type": "application/octet-stream"}, timeout=30,
        )
        response.raise_for_status()
        cloud_id = response.json()["id"]
        assert response.json()["point_count"] == 8192
        with openai.OpenAI(base_url=base_url + "/v1", api_key="local", timeout=180, max_retries=0) as client:
            assert client.models.list().data[0].id == "pointllm"
            messages = [{"role": "user", "content": "Describe this object, including its shape and color."}]
            started = time.monotonic()
            answer = client.chat.completions.create(
                model="pointllm", messages=messages, max_completion_tokens=64,
                temperature=0, extra_body={"point_cloud_id": cloud_id},
            )
            report["openai"] = {
                "text": answer.choices[0].message.content,
                "finish_reason": answer.choices[0].finish_reason,
                "usage": answer.usage.model_dump(),
                "seconds": round(time.monotonic() - started, 3),
            }
            assert report["openai"]["text"].strip()
            assert answer.usage.prompt_tokens >= 513
            assert 0 < answer.usage.completion_tokens <= 64
            messages.extend([
                {"role": "assistant", "content": answer.choices[0].message.content},
                {"role": "user", "content": "What is its main color?"},
            ])
            followup = client.chat.completions.create(
                model="pointllm", messages=messages, max_tokens=32, temperature=0,
                extra_body={"point_cloud_id": cloud_id},
            )
            report["followup"] = followup.choices[0].message.content
            assert report["followup"].strip()
            started = time.monotonic()
            first_content = None
            content = []
            final_reason = None
            usage = None
            with client.chat.completions.create(
                model="pointllm", messages=messages[:1], max_completion_tokens=64,
                temperature=0, extra_body={"point_cloud_id": cloud_id},
                stream=True, stream_options={"include_usage": True},
            ) as stream:
                for chunk in stream:
                    if chunk.usage:
                        usage = chunk.usage.model_dump()
                    for choice in chunk.choices:
                        if choice.delta.content:
                            if first_content is None:
                                first_content = time.monotonic() - started
                            content.append(choice.delta.content)
                        if choice.finish_reason:
                            final_reason = choice.finish_reason
            elapsed = time.monotonic() - started
            assert "".join(content).strip()
            assert len(content) > 1
            assert first_content < elapsed
            assert final_reason in ("stop", "length")
            assert usage and usage["completion_tokens"] > 0
            report["openai_stream"] = {
                "text": "".join(content), "content_chunks": len(content),
                "first_content_seconds": round(first_content, 3),
                "seconds": round(elapsed, 3), "finish_reason": final_reason, "usage": usage,
            }

        # Exercise the real backend with current LangChain in its isolated env.
        client_python = os.getenv("POINTLLM_CLIENT_PYTHON")
        if client_python:
            result = subprocess.run([
                client_python, "-c", """
import json, sys
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="pointllm", base_url=sys.argv[1] + "/v1", api_key="local",
    temperature=0, max_tokens=32, timeout=180, max_retries=0, use_responses_api=False,
).bind(extra_body={"point_cloud_id": sys.argv[2]})
answer = llm.invoke("Describe this object.")
assert answer.content.strip(), answer
followup = llm.invoke([("human", "Describe this object."), answer, ("human", "What color is it?")])
assert followup.content.strip(), followup
chunks = list(llm.stream("Describe this object."))
streamed = "".join(chunk.content for chunk in chunks)
assert streamed.strip(), chunks
print(json.dumps({"text": answer.content, "followup": followup.content, "stream_text": streamed,
                  "stream_chunks": len(chunks), "usage": answer.usage_metadata}))
""", base_url, cloud_id,
            ], check=True, capture_output=True, text=True, timeout=540)
            report["langchain"] = json.loads(result.stdout)
        else:
            report["langchain"] = "not tested: set POINTLLM_CLIENT_PYTHON"

        # Real-backend preflight errors must precede SSE headers.
        overflow = httpx.post(
            base_url + "/v1/chat/completions",
            json=chat_body(cloud_id, max_completion_tokens=2048, stream=True), timeout=30,
        )
        assert overflow.status_code == 400
        assert overflow.json()["error"]["code"] == "context_length_exceeded"
        assert httpx.delete(base_url + "/v1/point-clouds/" + cloud_id).json()["deleted"]

    report["status"] = "passed"
    destination = os.getenv("POINTLLM_TEST_REPORT")
    if destination:
        Path(destination).write_text(json.dumps(report, indent=2) + "\n")
    print("\nPRETRAINED_ENDPOINT_REPORT " + json.dumps(report), flush=True)

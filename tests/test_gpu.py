"""Opt-in backend smoke tests; tiny random weights are not a trained-model test."""

import os
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pointllm.serve.app import create_app
from pointllm.serve.config import Settings
from conftest import chat_body


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    configured = os.getenv("POINTLLM_TEST_MODEL")
    tiny = os.getenv("POINTLLM_TEST_TINY_GPU") == "1"
    if not configured and not tiny:
        pytest.skip("Set POINTLLM_TEST_MODEL to local weights or POINTLLM_TEST_TINY_GPU=1")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU smoke tests")
    if configured:
        assert Path(configured).is_dir(), "POINTLLM_TEST_MODEL must be an existing local checkpoint"
        return configured

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from pointllm.model import PointLLMConfig, PointLLMLlamaForCausalLM

    directory = tmp_path_factory.mktemp("tiny-pointllm")
    vocab = {
        "<unk>": 0, "<s>": 1, "</s>": 2, "<point_patch>": 3,
        "<point_start>": 4, "<point_end>": 5, "chair": 6,
    }
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        pad_token="</s>", additional_special_tokens=["<point_patch>", "<point_start>", "<point_end>"],
    )
    tokenizer.save_pretrained(directory)
    config = PointLLMConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=2048,
        point_backbone="PointBERT", use_color=True, mm_use_point_start_end=True,
        point_backbone_config_name="PointTransformer_8192point_2layer",
        DEFAULT_POINT_PATCH_TOKEN="<point_patch>", DEFAULT_POINT_START_TOKEN="<point_start>",
        DEFAULT_POINT_END_TOKEN="<point_end>", bos_token_id=1, eos_token_id=2,
    )
    torch.manual_seed(42)
    model = PointLLMLlamaForCausalLM(config)
    # Force predictable text while still executing the real encoder + LLaMA.
    model.generation_config.suppress_tokens = [index for index in vocab.values() if index != 6]
    model.generation_config._from_model_config = False
    model.save_pretrained(directory)
    return str(directory)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_real_backend_upload_chat_followup_stream(checkpoint, cloud_bytes, dtype):
    settings = Settings(model_path=checkpoint, dtype=dtype, request_timeout_seconds=180)
    with TestClient(create_app(settings)) as client:
        cloud_id = client.post(
            "/v1/point-clouds", content=cloud_bytes,
            headers={"Content-Type": "application/octet-stream"},
        ).json()["id"]
        response = client.post("/v1/chat/completions", json=chat_body(cloud_id, max_tokens=4))
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["usage"]["completion_tokens"] > 0
        answer = result["choices"][0]["message"]["content"]
        assert answer
        history = [
            {"role": "user", "content": "Describe it."},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "What color?"},
        ]
        assert client.post("/v1/chat/completions", json=chat_body(
            cloud_id, messages=history, max_tokens=4
        )).status_code == 200
        streamed = client.post("/v1/chat/completions", json=chat_body(
            cloud_id, max_tokens=4, stream=True, stream_options={"include_usage": True}
        ))
        assert streamed.status_code == 200
        assert "data: [DONE]" in streamed.text
        assert '"finish_reason": "length"' in streamed.text or '"finish_reason": "stop"' in streamed.text
        assert '"error"' not in streamed.text
        chunks = [
            json.loads(line[6:]) for line in streamed.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        streamed_text = "".join(
            choice["delta"].get("content", "")
            for chunk in chunks for choice in chunk["choices"]
        )
        if os.getenv("POINTLLM_TEST_TINY_GPU") == "1" and not os.getenv("POINTLLM_TEST_MODEL"):
            assert streamed_text == answer

import threading
import time
from types import SimpleNamespace

import pytest

from pointllm.serve.backend import IncrementalDecoder, PointLLMBackend
from pointllm.serve.config import Settings
from pointllm.serve.errors import APIError
from pointllm.serve.schemas import ChatRequest
from conftest import chat_body


class TextTokenizer:
    def decode(self, values, **kwargs):
        return "".join(values)


def test_incremental_decoder_holds_partial_words_and_stop():
    chunks = []
    decoder = IncrementalDecoder(TextTokenizer(), "</s>", chunks.append)
    decoder.update(list("A woo"))
    assert chunks == ["A "]
    decoder.update(list("A wooden chair.</"))
    assert chunks == ["A ", "wooden "]
    text = decoder.update(list("A wooden chair.</s>"), final=True)
    assert text == "A wooden chair."
    assert "".join(chunks) == text


def test_incremental_decoder_streams_chinese_without_spaces():
    chunks = []
    decoder = IncrementalDecoder(TextTokenizer(), "</s>", chunks.append)
    decoder.update(list("这是"))
    assert chunks == ["这是"]
    decoder.update(list("这是椅子"), final=True)
    assert "".join(chunks) == "这是椅子"


def test_prompt_construction_and_context_limit(monkeypatch, cloud_array):
    # No model weights or Torch installation are needed to verify prompt wiring.
    import sys
    captured = []

    class IDs:
        shape = (1, 530)

        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, text, **kwargs):
            captured.append(text)
            return SimpleNamespace(input_ids=IDs())

    class Tensor:
        def unsqueeze(self, dim):
            return self

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(tensor=lambda *a, **kw: Tensor()))
    backend = PointLLMBackend(Settings())
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(device="cpu", dtype="float32")
    backend.context_limit = 2048
    backend.point_config = {
        "default_point_patch_token": "<point_patch>", "point_token_len": 513,
        "mm_use_point_start_end": True,
        "default_point_start_token": "<point_start>",
        "default_point_end_token": "<point_end>",
    }
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Describe it."},
        {"role": "assistant", "content": "A chair."},
        {"role": "user", "content": "What color?"},
    ]
    request = ChatRequest(**chat_body("test", messages=messages))
    prepared = backend.prepare(request, cloud_array)
    prompt = captured[0]
    assert prompt.startswith("Be brief.")
    assert prompt.count("<point_patch>") == 513
    assert prompt.count("<point_start>") == 1
    assert prompt.endswith("ASSISTANT:")
    assert prepared.prompt_tokens == 530
    assert prepared.max_new_tokens == 256
    fresh = ChatRequest(**chat_body("test"))
    backend.prepare(fresh, cloud_array)
    assert "What color?" not in captured[1]
    with pytest.raises(APIError, match="exceeds"):
        backend.prepare(ChatRequest(**chat_body("test", max_tokens=2000)), cloud_array)
    with pytest.raises(APIError, match="reserved"):
        backend.prepare(ChatRequest(**chat_body(
            "test", messages=[{"role": "user", "content": "<point_patch>"}]
        )), cloud_array)


def test_legacy_generation_callback_streams_before_generate_returns():
    import sys
    from contextlib import nullcontext
    from unittest.mock import patch
    from pointllm.serve.backend import PreparedRequest

    state = {"generating": False}
    chunks = []

    class IDs:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, index):
            _, selection = index
            return IDs(self.values[selection])

        def tolist(self):
            return self.values

    class Model:
        generation_config = SimpleNamespace(eos_token_id=999)

        def generate(self, input_ids, **options):
            assert not options["do_sample"]
            assert "temperature" not in options
            state["generating"] = True
            for text in ("A ", "A wooden ", "A wooden chair."):
                options["stopping_criteria"][0](IDs(["p"] + list(text)), None)
            state["generating"] = False
            return IDs(["p"] + list("A wooden chair."))

    def emit(text):
        chunks.append((text, state["generating"]))

    backend = PointLLMBackend(Settings())
    backend.model = Model()
    backend.tokenizer = TextTokenizer()
    backend.tokenizer.eos_token_id = 999
    prepared = PreparedRequest(None, None, 1, 100, 0, 0.95, "</s>")
    with patch.dict(sys.modules, {
        "torch": SimpleNamespace(inference_mode=nullcontext),
        "transformers": SimpleNamespace(StoppingCriteria=object),
    }):
        result = backend.generate(prepared, threading.Event(), time.monotonic() + 5, emit)
    assert result.text == "A wooden chair."
    assert "".join(text for text, _ in chunks) == result.text
    assert any(during_generation for _, during_generation in chunks)

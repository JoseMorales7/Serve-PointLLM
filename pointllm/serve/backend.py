import time
from dataclasses import dataclass
from typing import Any

from .errors import APIError


@dataclass
class PreparedRequest:
    input_ids: Any
    point_clouds: Any
    prompt_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    stop: str


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str

    def usage(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


def check_cancel(cancel, deadline):
    if time.monotonic() >= deadline:
        raise APIError("Generation deadline exceeded", 504, "request_timeout")
    if cancel.is_set():
        raise APIError("Generation cancelled", 499, "request_cancelled")


class IncrementalDecoder:
    """Publish complete words without exposing incomplete Unicode/stop suffixes."""

    def __init__(self, tokenizer, stop, emit):
        self.tokenizer = tokenizer
        self.stop = stop
        self.emit = emit
        self.published = ""

    def update(self, token_ids, final=False):
        text = self.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if self.stop and self.stop in text:
            text = text.split(self.stop, 1)[0]
        if final:
            stable = text
        else:
            # Hold a possible partial textual stop marker.
            held = 0
            for count in range(1, min(len(text), len(self.stop)) + 1):
                if text.endswith(self.stop[:count]):
                    held = count
            candidate = text[:-held] if held else text
            boundary = max(candidate.rfind(" "), candidate.rfind("\n")) + 1
            if candidate and (
                0x4E00 <= ord(candidate[-1]) <= 0x9FFF
                or 0x3400 <= ord(candidate[-1]) <= 0x4DBF
                or 0x20000 <= ord(candidate[-1]) <= 0x2FA1F
            ):
                boundary = len(candidate)
            stable = candidate[:boundary]
        if not stable.startswith(self.published):
            raise RuntimeError("Tokenizer changed an already streamed prefix")
        delta = stable[len(self.published):]
        if delta:
            self.emit(delta)
            self.published = stable
        return text


class PointLLMBackend:
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.tokenizer = None

    def load(self):
        import torch
        from transformers import AutoTokenizer
        from pointllm.model import PointLLMLlamaForCausalLM

        if self.settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; configure POINTLLM_DEVICE explicitly")
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_path)
        self.model = PointLLMLlamaForCausalLM.from_pretrained(
            self.settings.model_path,
            low_cpu_mem_usage=False,
            use_cache=True,
            torch_dtype=getattr(torch, self.settings.dtype),
        ).to(self.settings.device)
        self.model.initialize_tokenizer_point_backbone_config_wo_embedding(self.tokenizer)
        self.model.eval()
        self.point_config = self.model.get_model().point_backbone_config
        if self.point_config["point_cloud_dim"] != 6:
            raise RuntimeError("The serving upload contract requires a six-channel XYZRGB checkpoint")
        self.context_limit = min(
            self.settings.max_context_tokens,
            getattr(self.model.config, "max_position_embeddings", 2048),
        )

    def prepare(self, request, cloud):
        import torch
        from pointllm.conversation import conv_templates, SeparatorStyle

        conv = conv_templates["vicuna_v1_1"].copy()
        config = self.point_config
        prefix = config["default_point_patch_token"] * config["point_token_len"]
        if config["mm_use_point_start_end"]:
            prefix = config["default_point_start_token"] + prefix + config["default_point_end_token"]
        first_user = True
        # Reject client-supplied reserved tokens: they can alter feature placement.
        reserved = [config["default_point_patch_token"]]
        if config["mm_use_point_start_end"]:
            reserved += [config["default_point_start_token"], config["default_point_end_token"]]
        for message in request.messages:
            if any(token in message.content for token in reserved):
                raise APIError("Messages cannot contain reserved point tokens", param="messages")
            if message.role == "system":
                conv.system = message.content
                continue
            content = message.content
            if first_user and message.role == "user":
                content = prefix + "\n" + content
                first_user = False
            conv.append_message(conv.roles[0 if message.role == "user" else 1], content)
        conv.append_message(conv.roles[1], None)
        input_ids = self.tokenizer(conv.get_prompt(), return_tensors="pt").input_ids
        prompt_tokens = input_ids.shape[1]
        if prompt_tokens + request.output_limit > self.context_limit:
            raise APIError(
                f"Prompt ({prompt_tokens}) plus output ({request.output_limit}) exceeds "
                f"the {self.context_limit}-token context limit",
                code="context_length_exceeded", param="messages",
            )
        return PreparedRequest(
            input_ids.to(self.model.device),
            torch.tensor(cloud, device=self.model.device, dtype=self.model.dtype).unsqueeze(0),
            prompt_tokens, request.output_limit, request.temperature, request.top_p,
            conv.sep2 if conv.sep_style == SeparatorStyle.TWO else conv.sep,
        )

    def generate(self, prepared, cancel, deadline, emit):
        import torch
        from transformers import StoppingCriteria

        check_cancel(cancel, deadline)
        decoder = IncrementalDecoder(self.tokenizer, prepared.stop, emit)
        tokenizer = self.tokenizer
        stopped_on_separator = False

        class Monitor(StoppingCriteria):
            def __call__(self, output_ids, scores, **kwargs):
                nonlocal stopped_on_separator
                if cancel.is_set() or time.monotonic() >= deadline:
                    return True
                tokens = output_ids[0, prepared.prompt_tokens:].tolist()
                decoder.update(tokens)
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                stopped_on_separator = bool(prepared.stop and prepared.stop in text)
                return stopped_on_separator

        options = {
            "max_new_tokens": prepared.max_new_tokens,
            "do_sample": prepared.temperature > 0,
            "stopping_criteria": [Monitor()],
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if prepared.temperature > 0:
            options.update(temperature=prepared.temperature, top_p=prepared.top_p)
        with torch.inference_mode():
            output_ids = self.model.generate(
                prepared.input_ids, point_clouds=prepared.point_clouds, **options
            )
        check_cancel(cancel, deadline)
        generated = output_ids[0, prepared.prompt_tokens:].tolist()
        text = decoder.update(generated, final=True)
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        ended = stopped_on_separator or (generated and generated[-1] in eos_ids)
        return GenerationResult(
            text, prepared.prompt_tokens, len(generated),
            "stop" if ended or len(generated) < prepared.max_new_tokens else "length",
        )

    def close(self):
        self.model = None
        self.tokenizer = None

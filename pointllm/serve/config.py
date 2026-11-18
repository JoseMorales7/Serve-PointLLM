import os
from dataclasses import dataclass, fields
from typing import Optional


@dataclass(frozen=True)
class Settings:
    model_path: str = "RunsenXu/PointLLM_7B_v1.2"
    model_alias: str = "pointllm"
    device: str = "cuda:0"
    dtype: str = "float16"
    api_key: Optional[str] = None
    max_context_tokens: int = 2048
    max_upload_bytes: int = 16 * 1024 * 1024
    max_request_bytes: int = 1024 * 1024
    max_input_points: int = 32768
    max_clouds: int = 32
    cloud_ttl_seconds: float = 3600
    max_pending_requests: int = 4
    request_timeout_seconds: float = 120

    def __post_init__(self):
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("POINTLLM_DTYPE must be float16, bfloat16, or float32")
        if not self.model_path or not self.model_alias or not self.device:
            raise ValueError("Model path, alias, and device cannot be empty")
        for name in (
            "max_context_tokens", "max_upload_bytes", "max_request_bytes", "max_input_points",
            "max_clouds", "cloud_ttl_seconds", "max_pending_requests",
            "request_timeout_seconds",
        ):
            value = getattr(self, name)
            if not 0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
        if self.max_input_points < 8192:
            raise ValueError("max_input_points must be at least 8192")

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for field in fields(cls):
            raw = os.getenv("POINTLLM_" + field.name.upper())
            if raw is not None:
                default = getattr(defaults, field.name)
                values[field.name] = type(default)(raw) if default is not None else raw
        return cls(**values)

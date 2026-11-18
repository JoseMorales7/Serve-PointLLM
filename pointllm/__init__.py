"""Keep HTTP tooling importable without importing the GPU model stack."""

__all__ = ["PointLLMLlamaForCausalLM"]


def __getattr__(name):
    if name == "PointLLMLlamaForCausalLM":
        from .model import PointLLMLlamaForCausalLM

        return PointLLMLlamaForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

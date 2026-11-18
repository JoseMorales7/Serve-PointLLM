import math
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class StreamOptions(StrictModel):
    include_usage: bool = False


class ChatRequest(StrictModel):
    model: str
    messages: List[Message] = Field(min_length=1)
    point_cloud_id: str
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    max_completion_tokens: Optional[int] = Field(default=None, gt=0)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    n: Literal[1] = 1

    @model_validator(mode="after")
    def validate_conversation(self):
        if not math.isfinite(self.temperature) or not math.isfinite(self.top_p):
            raise ValueError("Sampling parameters must be finite")
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            if self.max_tokens != self.max_completion_tokens:
                raise ValueError("Conflicting max_tokens and max_completion_tokens")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        dialogue = self.messages
        if dialogue[0].role == "system":
            dialogue = dialogue[1:]
        if not dialogue or dialogue[-1].role != "user":
            raise ValueError("Conversation must end with a user message")
        for index, message in enumerate(dialogue):
            expected = "user" if index % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError("Use an optional leading system message, then alternating user/assistant messages")
        return self

    @property
    def output_limit(self):
        return self.max_completion_tokens or self.max_tokens or 256

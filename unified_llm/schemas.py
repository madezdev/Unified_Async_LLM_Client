"""Validated input and output models for the LLM clients."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    provider: Literal["openai", "anthropic"]
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, gt=0)

    @model_validator(mode="after")
    def validate_provider_temperature(self) -> "ModelConfig":
        if self.provider == "anthropic" and self.temperature > 1:
            raise ValueError("Anthropic requiere una temperatura entre 0 y 1")
        return self


class LLMError(BaseModel):
    code: Literal["authentication", "rate_limit", "network", "api_error"]
    message: str
    retryable: bool = False


class ModelResponse(BaseModel):
    provider: Literal["openai", "anthropic"]
    model: str
    content: str = ""
    error: LLMError | None = None


class StreamEvent(BaseModel):
    text: str = ""
    error: LLMError | None = None

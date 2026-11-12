from collections.abc import AsyncIterator, Sequence

import openai
from openai import AsyncOpenAI

from ..base import BaseLLMClient
from ..errors import error_from_exception
from ..schemas import ChatMessage, ModelConfig, ModelResponse, StreamEvent


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        if config.provider != "openai":
            raise ValueError("OpenAIClient requiere provider='openai'")
        super().__init__(config)
        self._client = AsyncOpenAI(api_key=api_key)

    def _parameters(self, messages: Sequence[ChatMessage]) -> dict:
        return {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [message.model_dump() for message in messages],
        }

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(**self._parameters(messages))
            content = response.choices[0].message.content or ""
            return ModelResponse(provider="openai", model=self.config.model, content=content)
        except openai.APIError as exc:
            return ModelResponse(provider="openai", model=self.config.model, error=error_from_exception(exc))

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        try:
            response = await self._client.chat.completions.create(
                **self._parameters(messages), stream=True
            )
            async for chunk in response:
                if chunk.choices:
                    text = chunk.choices[0].delta.content
                    if text:
                        yield StreamEvent(text=text)
        except openai.APIError as exc:
            yield StreamEvent(error=error_from_exception(exc))

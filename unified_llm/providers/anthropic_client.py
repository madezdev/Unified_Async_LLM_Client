from collections.abc import AsyncIterator, Sequence

import anthropic
from anthropic import AsyncAnthropic

from ..base import BaseLLMClient
from ..errors import error_from_exception
from ..schemas import ChatMessage, ModelConfig, ModelResponse, StreamEvent


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        if config.provider != "anthropic":
            raise ValueError("AnthropicClient requiere provider='anthropic'")
        super().__init__(config)
        self._client = AsyncAnthropic(api_key=api_key)

    def _parameters(self, messages: Sequence[ChatMessage]) -> dict:
        system = "\n\n".join(message.content for message in messages if message.role == "system")
        conversation = [message.model_dump() for message in messages if message.role != "system"]
        parameters = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": conversation,
        }
        if system:
            parameters["system"] = system
        return parameters

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.messages.create(**self._parameters(messages))
            content = "".join(block.text for block in response.content if block.type == "text")
            return ModelResponse(provider="anthropic", model=self.config.model, content=content)
        except anthropic.APIError as exc:
            return ModelResponse(provider="anthropic", model=self.config.model, error=error_from_exception(exc))

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        try:
            async with self._client.messages.stream(**self._parameters(messages)) as response:
                async for text in response.text_stream:
                    if text:
                        yield StreamEvent(text=text)
        except anthropic.APIError as exc:
            yield StreamEvent(error=error_from_exception(exc))

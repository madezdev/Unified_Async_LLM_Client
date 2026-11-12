from collections.abc import AsyncIterator, Sequence

from .base import BaseLLMClient
from .providers import AnthropicClient, OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, StreamEvent


class AsyncLLMManager:
    """Select a provider behind the same generate/stream API."""

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        if not api_key:
            raise ValueError("Se requiere la clave de API del proveedor")
        self.client: BaseLLMClient = (
            OpenAIClient(config, api_key)
            if config.provider == "openai"
            else AnthropicClient(config, api_key)
        )

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        return self.client.stream(messages)

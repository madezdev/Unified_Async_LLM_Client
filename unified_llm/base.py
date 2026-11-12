from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from .schemas import ChatMessage, ModelConfig, ModelResponse, StreamEvent


class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Return a complete response or a structured API error."""

    @abstractmethod
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        """Yield text fragments or one structured error event."""

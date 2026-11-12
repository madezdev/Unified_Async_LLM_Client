import unittest
from types import SimpleNamespace

import httpx
import openai
from pydantic import ValidationError

from unified_llm import AsyncLLMManager, ChatMessage, ModelConfig
from unified_llm.providers import AnthropicClient, OpenAIClient
from unified_llm.settings import manager_from_env


class SchemaTests(unittest.TestCase):
    def test_config_limits_and_message_validation(self) -> None:
        with self.assertRaises(ValidationError):
            ModelConfig(provider="openai", model="example", temperature=2.1)
        with self.assertRaises(ValidationError):
            ModelConfig(provider="anthropic", model="example", temperature=1.1)
        with self.assertRaises(ValidationError):
            ModelConfig(provider="openai", model="example", max_tokens=0)
        with self.assertRaises(ValidationError):
            ChatMessage(role="user", content="")

    def test_manager_selects_provider(self) -> None:
        openai_manager = AsyncLLMManager(ModelConfig(provider="openai", model="m"), "key")
        anthropic_manager = AsyncLLMManager(
            ModelConfig(provider="anthropic", model="m"), "key"
        )
        self.assertIsInstance(openai_manager.client, OpenAIClient)
        self.assertIsInstance(anthropic_manager.client, AnthropicClient)


class OpenAIClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_and_stream(self) -> None:
        calls = []

        async def chunks():
            for part in ("Ho", "la"):
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=part))]
                )

        async def create(**kwargs):
            calls.append(kwargs)
            if kwargs.get("stream"):
                return chunks()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Hola"))]
            )

        client = OpenAIClient(ModelConfig(provider="openai", model="test"), "key")
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        messages = [ChatMessage(role="user", content="Saluda")]

        response = await client.generate(messages)
        streamed = [event async for event in client.stream(messages)]

        self.assertEqual(response.content, "Hola")
        self.assertEqual([event.text for event in streamed], ["Ho", "la"])
        self.assertEqual(calls[0]["messages"], [{"role": "user", "content": "Saluda"}])
        self.assertTrue(calls[1]["stream"])

    async def test_rate_limit_is_controlled_in_both_modes(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        error = openai.RateLimitError(
            "Too many requests", response=httpx.Response(429, request=request), body=None
        )

        async def create(**kwargs):
            raise error

        client = OpenAIClient(ModelConfig(provider="openai", model="test"), "key")
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        messages = [ChatMessage(role="user", content="Hola")]

        response = await client.generate(messages)
        streamed = [event async for event in client.stream(messages)]

        self.assertEqual(response.error.code, "rate_limit")
        self.assertTrue(response.error.retryable)
        self.assertEqual(streamed[0].error.code, "rate_limit")

    async def test_connection_error_is_controlled(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

        async def create(**kwargs):
            raise openai.APIConnectionError(request=request)

        client = OpenAIClient(ModelConfig(provider="openai", model="test"), "key")
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        messages = [ChatMessage(role="user", content="Hola")]

        response = await client.generate(messages)
        streamed = [event async for event in client.stream(messages)]

        self.assertEqual(response.error.code, "network")
        self.assertTrue(response.error.retryable)
        self.assertEqual(streamed[0].error.code, "network")


class AnthropicClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_system_messages_and_stream(self) -> None:
        calls = []

        async def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Hola")])

        class FakeStream:
            async def __aenter__(self):
                async def parts():
                    yield "Ho"
                    yield "la"

                return SimpleNamespace(text_stream=parts())

            async def __aexit__(self, *args):
                return False

        def stream(**kwargs):
            calls.append(kwargs)
            return FakeStream()

        client = AnthropicClient(ModelConfig(provider="anthropic", model="test"), "key")
        client._client = SimpleNamespace(messages=SimpleNamespace(create=create, stream=stream))
        messages = [
            ChatMessage(role="system", content="Responde en español"),
            ChatMessage(role="user", content="Saluda"),
        ]

        response = await client.generate(messages)
        streamed = [event async for event in client.stream(messages)]

        self.assertEqual(response.content, "Hola")
        self.assertEqual([event.text for event in streamed], ["Ho", "la"])
        self.assertEqual(calls[0]["system"], "Responde en español")
        self.assertEqual(calls[0]["messages"], [{"role": "user", "content": "Saluda"}])


if __name__ == "__main__":
    unittest.main()

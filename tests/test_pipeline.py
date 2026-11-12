import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

import httpx
import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from entity_pipeline import (
    CriticalityLevel,
    InvalidOutputError,
    ProviderError,
    RefusedOutputError,
    TechnicalExtraction,
    TruncatedOutputError,
    build_chain,
    build_prompt,
    process_text,
)
from entity_pipeline.chain import _structured_output_options, build_chat_model
from unified_llm import ModelConfig
from unified_llm.settings import config_from_env

VALID_ARGS = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "La API agota el pool de conexiones de PostgreSQL y responde 503.",
}
LOGGER = "entity_pipeline.chain"


class FakeToolModel(GenericFakeChatModel):
    """Returns the queued messages in order; tool binding is a no-op."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolModel":
        return self


class FailingModel(FakeToolModel):
    error: Exception

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise self.error


def reply(args: dict[str, Any] = VALID_ARGS, **metadata: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "TechnicalExtraction", "args": args, "id": "call_1"}],
        response_metadata=metadata or {"finish_reason": "stop"},
    )


def chain_with(*replies: AIMessage):
    return build_chain(FakeToolModel(messages=iter(replies)), wait_between_attempts=False)


class SchemaTests(unittest.TestCase):
    def test_technologies_are_trimmed_and_deduplicated(self) -> None:
        result = TechnicalExtraction(
            **{**VALID_ARGS, "tecnologias": [" Redis ", "redis", "", "Apache  Kafka"]}
        )
        self.assertEqual(result.tecnologias, ["Redis", "Apache Kafka"])
        self.assertEqual(result.nivel_de_criticidad, CriticalityLevel.ALTA)

    def test_invalid_values_are_rejected(self) -> None:
        for field, value in [
            ("tecnologias", []),
            ("tecnologias", ["  "]),
            ("tecnologias", [f"Tecnologia {n}" for n in range(31)]),
            ("tecnologias", ["x" * 61]),
            ("nivel_de_criticidad", "critica"),
            ("resumen_tecnico", "corto"),
            ("resumen_tecnico", "x" * 401),
        ]:
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                TechnicalExtraction(**{**VALID_ARGS, field: value})

    def test_control_characters_are_removed(self) -> None:
        result = TechnicalExtraction(
            **{**VALID_ARGS, "tecnologias": ["\x1b[31mRedis\x1b[0m", "Apache\nKafka"]}
        )
        self.assertEqual(result.tecnologias, ["[31mRedis[0m", "Apache Kafka"])
        self.assertNotIn("\x1b", result.model_dump_json())

    def test_output_serializes_to_expected_json(self) -> None:
        dumped = TechnicalExtraction(**VALID_ARGS).model_dump(mode="json")
        self.assertEqual(dumped, VALID_ARGS)


class PromptTests(unittest.TestCase):
    def test_prompt_receives_text_and_format_instructions(self) -> None:
        prompt = build_prompt("REGLAS DE PRUEBA")
        system, human = prompt.format_messages(text="log {con llaves}")
        tag = re.match(r"<(texto_[0-9a-f]{8})>\nlog \{con llaves\}\n</\1>\n", human.content)
        self.assertIsNotNone(tag, human.content)
        self.assertIn("REGLAS DE PRUEBA", system.content)
        self.assertIn(f"<{tag.group(1)}>", system.content)
        _, other = prompt.format_messages(text="log")
        self.assertNotIn(tag.group(1), other.content)


class ModelSetupTests(unittest.TestCase):
    def test_env_file_config_is_reused_with_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / ".env"
            env.write_text("LLM_PROVIDER=anthropic\nLLM_MODEL=modelo\nANTHROPIC_API_KEY=clave\n")
            config, key = config_from_env(env, temperature=0, max_tokens=64)
        self.assertEqual(
            (config.provider, config.model, config.max_tokens, key),
            ("anthropic", "modelo", 64, "clave"),
        )

    def test_provider_selects_client_and_structured_output_method(self) -> None:
        openai_model = build_chat_model(
            ModelConfig(provider="openai", model="m", temperature=0), "key"
        )
        anthropic_model = build_chat_model(
            ModelConfig(provider="anthropic", model="m", temperature=0), "key"
        )
        self.assertIsInstance(openai_model, ChatOpenAI)
        self.assertIsInstance(anthropic_model, ChatAnthropic)
        self.assertEqual(
            _structured_output_options(openai_model), {"method": "function_calling", "strict": True}
        )
        self.assertEqual(
            _structured_output_options(anthropic_model), {"method": "function_calling"}
        )

    def test_missing_api_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_chat_model(ModelConfig(provider="openai", model="m"), "")


class ProcessTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_validated_object(self) -> None:
        result = await process_text("ERROR pool exhausted", chain=chain_with(reply()))
        self.assertEqual(result.model_dump(mode="json"), VALID_ARGS)

    async def test_truncated_response_is_retried_even_if_it_parses(self) -> None:
        for metadata in ({"finish_reason": "length"}, {"stop_reason": "max_tokens"}):
            with self.subTest(metadata=metadata), self.assertLogs(LOGGER, level="INFO") as logs:
                chain = chain_with(reply(**metadata), reply())
                result = await process_text("ERROR pool exhausted", chain=chain)
            self.assertEqual(result.tecnologias, VALID_ARGS["tecnologias"])
            self.assertTrue(any("límite de tokens" in line for line in logs.output))
            self.assertTrue(any("Intento 2" in line for line in logs.output))

    async def test_malformed_output_is_retried(self) -> None:
        missing_field = {"tecnologias": ["Redis"], "nivel_de_criticidad": "alta"}
        result = await process_text("texto", chain=chain_with(reply(missing_field), reply()))
        self.assertEqual(result.nivel_de_criticidad, CriticalityLevel.ALTA)

    async def test_gives_up_after_max_attempts(self) -> None:
        empty = {**VALID_ARGS, "tecnologias": []}
        chain = chain_with(reply(empty), reply(empty), reply(empty))
        with self.assertLogs(LOGGER, level="ERROR") as logs, self.assertRaises(InvalidOutputError):
            await process_text("texto", chain=chain)
        self.assertIn("tras 3 intento(s)", logs.output[-1])

    async def test_still_truncated_after_retries_raises(self) -> None:
        chain = chain_with(*(reply(finish_reason="length") for _ in range(3)))
        with self.assertRaises(TruncatedOutputError):
            await process_text("texto", chain=chain)

    async def test_refusal_is_not_retried(self) -> None:
        # A single queued reply: a retry would exhaust the fake model and fail differently.
        with self.assertRaises(RefusedOutputError):
            await process_text("texto", chain=chain_with(reply(stop_reason="refusal")))

    async def test_invalid_input_never_reaches_the_model(self) -> None:
        for text in ("", "   \n", "x" * 20_001):
            with self.subTest(length=len(text)), self.assertRaises(ValueError):
                await process_text(text, chain=chain_with())

    async def test_provider_errors_are_mapped(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        error = openai.RateLimitError(
            "Too many requests", response=httpx.Response(429, request=request), body=None
        )
        chain = build_chain(
            FailingModel(messages=iter([]), error=error), wait_between_attempts=False
        )
        with self.assertRaises(ProviderError) as caught:
            await process_text("texto", chain=chain)
        self.assertEqual(caught.exception.code, "rate_limit")
        self.assertTrue(caught.exception.retryable)

    async def test_api_key_never_reaches_logs_or_errors(self) -> None:
        secret = "sk-test-no-debe-aparecer"
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        error = openai.AuthenticationError(
            f"Incorrect API key provided: {secret}",
            response=httpx.Response(401, request=request),
            body=None,
        )
        chain = build_chain(
            FailingModel(messages=iter([]), error=error), wait_between_attempts=False
        )
        with (
            self.assertLogs(LOGGER, level="INFO") as logs,
            self.assertRaises(ProviderError) as caught,
        ):
            await process_text("texto", chain=chain)
        self.assertNotIn(secret, str(caught.exception))
        self.assertFalse(any(secret in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()

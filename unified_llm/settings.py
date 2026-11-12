from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from .manager import AsyncLLMManager
from .schemas import ModelConfig


def config_from_env(path: Path, **overrides: Any) -> tuple[ModelConfig, str]:
    """Load the model config and API key only from the project's local, untracked .env file."""
    if not path.is_file():
        raise ValueError(f"Falta {path.name}; copia .env.example y agrega tu clave")
    values = dotenv_values(path)
    provider = (values.get("LLM_PROVIDER") or "openai").lower()
    model = values.get("LLM_MODEL") or (
        "gpt-4o-mini" if provider == "openai" else "claude-3-5-haiku-latest"
    )
    config = ModelConfig(provider=provider, model=model, **overrides)
    key_name = "OPENAI_API_KEY" if config.provider == "openai" else "ANTHROPIC_API_KEY"
    return config, values.get(key_name) or ""


def manager_from_env(path: Path) -> AsyncLLMManager:
    return AsyncLLMManager(*config_from_env(path))

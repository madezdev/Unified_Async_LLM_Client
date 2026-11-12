import logging

from .chain import build_chain, build_prompt, chain_from_env, process_text
from .errors import (
    ExtractionError,
    InvalidOutputError,
    ProviderError,
    RefusedOutputError,
    TruncatedOutputError,
)
from .schemas import CriticalityLevel, TechnicalExtraction

__all__ = [
    "CriticalityLevel",
    "ExtractionError",
    "InvalidOutputError",
    "ProviderError",
    "RefusedOutputError",
    "TechnicalExtraction",
    "TruncatedOutputError",
    "build_chain",
    "build_prompt",
    "chain_from_env",
    "process_text",
]

# The application decides where logs go; without this, warnings fall back to stderr.
logging.getLogger(__name__).addHandler(logging.NullHandler())

"""Errors raised when the pipeline cannot produce a valid object."""

from unified_llm.errors import error_from_exception


class ExtractionError(Exception):
    """No valid extraction was obtained."""


class TruncatedOutputError(ExtractionError):
    """The model stopped at the token limit, so the object may be incomplete."""


class InvalidOutputError(ExtractionError):
    """Malformed JSON, missing fields or values rejected by the schema."""


class RefusedOutputError(ExtractionError):
    """The model or the content filter refused to answer; retrying will not help."""


class ProviderError(ExtractionError):
    """The provider API failed after the SDK's own retries."""

    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable

    @classmethod
    def from_exception(cls, exc: Exception) -> "ProviderError":
        error = error_from_exception(exc)
        return cls(error.message, code=error.code, retryable=error.retryable)

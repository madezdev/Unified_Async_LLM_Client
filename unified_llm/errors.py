import anthropic
import openai

from .schemas import LLMError


def error_from_exception(exc: Exception) -> LLMError:
    if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        return LLMError(code="rate_limit", message="Límite de solicitudes alcanzado.", retryable=True)
    if isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)):
        return LLMError(code="authentication", message="La clave de API no es válida.")
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return LLMError(code="network", message="No se pudo conectar con el proveedor.", retryable=True)
    return LLMError(code="api_error", message="El proveedor rechazó la solicitud.")

"""Output contract of the pipeline: what the model must return and how it is validated."""

import unicodedata
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

SUMMARY_MIN_CHARS = 20
SUMMARY_MAX_CHARS = 400
MAX_TECHNOLOGIES = 30
MAX_TECHNOLOGY_CHARS = 60


def _clean(text: str) -> str:
    """Drop control and invisible characters (ANSI escapes, zero-width spaces) and collapse whitespace."""
    visible = "".join(
        char for char in text if char.isspace() or not unicodedata.category(char).startswith("C")
    )
    return " ".join(visible.split())


class CriticalityLevel(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class TechnicalExtraction(BaseModel):
    """Entidades técnicas extraídas de un log de error, un incidente o una descripción de arquitectura."""

    # The docstring and descriptions reach the model as the tool schema, hence Spanish. Limits
    # live in validators instead of Field(min_length=...) so that schema stays plain enough
    # for OpenAI strict mode and Anthropic tools.
    tecnologias: list[str] = Field(
        description=(
            "Tecnologías mencionadas en el texto (lenguajes, frameworks, bases de datos, "
            "servicios cloud, colas, herramientas), con su nombre oficial. Al menos una."
        )
    )
    nivel_de_criticidad: CriticalityLevel = Field(
        description="Impacto del problema o del riesgo que describe el texto."
    )
    resumen_tecnico: str = Field(description="Resumen técnico de una o dos oraciones, en español.")

    @field_validator("tecnologias")
    @classmethod
    def normalize_technologies(cls, names: list[str]) -> list[str]:
        unique: dict[str, str] = {}
        for name in names:
            clean = _clean(name)
            if any(char.isalnum() for char in clean):
                unique.setdefault(clean.casefold(), clean)
        if not unique:
            raise ValueError("debe incluir al menos una tecnología")
        if len(unique) > MAX_TECHNOLOGIES:
            raise ValueError(f"admite hasta {MAX_TECHNOLOGIES} tecnologías")
        if any(len(name) > MAX_TECHNOLOGY_CHARS for name in unique.values()):
            raise ValueError(f"cada tecnología admite hasta {MAX_TECHNOLOGY_CHARS} caracteres")
        return list(unique.values())

    @field_validator("nivel_de_criticidad", mode="before")
    @classmethod
    def normalize_criticality(cls, value: object) -> object:
        return value.strip().casefold() if isinstance(value, str) else value

    @field_validator("resumen_tecnico")
    @classmethod
    def check_summary_length(cls, summary: str) -> str:
        clean = _clean(summary)
        if not SUMMARY_MIN_CHARS <= len(clean) <= SUMMARY_MAX_CHARS:
            raise ValueError(
                f"debe tener entre {SUMMARY_MIN_CHARS} y {SUMMARY_MAX_CHARS} caracteres"
            )
        return clean

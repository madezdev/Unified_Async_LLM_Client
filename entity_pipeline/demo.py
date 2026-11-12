"""Async smoke test: runs the sample texts concurrently and prints each validated object."""

import argparse
import asyncio
import logging

from .chain import DEFAULT_MAX_TOKENS, chain_from_env, process_text
from .errors import ExtractionError

SAMPLES = {
    "log de error": (
        "2026-11-10 03:12:44 ERROR orders-api: psycopg.errors.ConnectionTimeout: couldn't get a "
        "connection after 30.00 sec (pool max_size=20) en POST /orders. FastAPI responde 503 a todos "
        "los clientes desde hace 15 minutos. Redis contesta normal, pero PostgreSQL acumula 480 "
        "conexiones en estado 'idle in transaction'."
    ),
    "arquitectura": (
        "El frontend en Next.js consume una API GraphQL escrita en Node.js. Los eventos de pedidos "
        "se publican en Kafka y un worker en Python los guarda en MongoDB. Todo corre en Kubernetes "
        "sobre AWS y el despliegue se hace con GitHub Actions. No hay incidentes abiertos."
    ),
    "texto ambiguo": (
        "Ayer a la tarde el sistema anduvo lento un rato y algunos usuarios se quejaron, pero después "
        "se normalizó solo. Puede que haya sido la base o la caché, nadie revisó los logs todavía."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae entidades técnicas de un texto.")
    parser.add_argument("texto", nargs="?", help="Texto a procesar; sin él se usan los ejemplos.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Límite de tokens de salida. Con un valor bajo (ej. 30) se ven los reintentos.",
    )
    return parser.parse_args()


async def run(texts: dict[str, str], max_tokens: int) -> int:
    chain = chain_from_env(max_tokens=max_tokens)
    results = await asyncio.gather(
        *(process_text(text, chain=chain) for text in texts.values()), return_exceptions=True
    )
    failures = 0
    for label, result in zip(texts, results, strict=True):
        print(f"\n### {label}")
        if isinstance(result, (ExtractionError, ValueError)):
            failures += 1
            print(f"Sin resultado válido: {result}")
        elif isinstance(result, BaseException):
            raise result
        else:
            print(result.model_dump_json(indent=2))
    return failures


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    texts = {"texto recibido": args.texto} if args.texto is not None else SAMPLES
    try:
        failures = asyncio.run(run(texts, args.max_tokens))
    except ValueError as exc:
        raise SystemExit(f"Configuración inválida: {exc}") from exc
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

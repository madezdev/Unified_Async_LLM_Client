"""Run a short normal and streaming request against the selected provider."""

import asyncio
import sys
from pathlib import Path

from pydantic import ValidationError

from unified_llm import ChatMessage
from unified_llm.settings import manager_from_env


async def main(prompt: str) -> None:
    try:
        manager = manager_from_env(Path(__file__).with_name(".env"))
    except (ValidationError, ValueError) as exc:
        print(f"Configuración inválida: {exc}")
        return

    messages = [ChatMessage(role="user", content=prompt)]
    response = await manager.generate(messages)
    print("Respuesta normal:")
    if response.error:
        print(f"Error ({response.error.code}): {response.error.message}")
    else:
        print(response.content)

    print("\nRespuesta en streaming:")
    async for event in manager.stream(messages):
        if event.error:
            print(f"\nError ({event.error.code}): {event.error.message}")
            break
        print(event.text, end="", flush=True)
    print()


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]).strip() or input("Escribí tu pregunta: ").strip()
    if prompt:
        asyncio.run(main(prompt))
    else:
        print("La pregunta no puede estar vacía.")

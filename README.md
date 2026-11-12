# Unified Async LLM Client

Cliente asíncrono para OpenAI y Anthropic con una interfaz común, validación con Pydantic y respuestas en streaming.

## Requisitos e instalación

Se requiere Python 3.12 y una clave de API del proveedor elegido.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Completa la clave correspondiente en `.env`. El archivo `.env` está excluido de Git.
Las credenciales se leen únicamente de ese archivo local, no de variables del sistema.

## Estructura

```text
unified_llm/
├── schemas.py          # Validación de mensajes, configuración y respuestas
├── settings.py         # Lectura del .env local
├── manager.py          # Selección del proveedor
├── base.py             # Contrato asíncrono común
├── errors.py           # Errores controlados
└── providers/          # Adaptadores OpenAI y Anthropic
main.py                 # Prueba manual
tests/                  # Pruebas unitarias
```

## Configuración

| Variable | Descripción |
| --- | --- |
| `LLM_PROVIDER` | `openai` o `anthropic` |
| `LLM_MODEL` | Nombre del modelo del proveedor; si se omite, se usa un modelo predeterminado |
| `OPENAI_API_KEY` | Necesaria cuando `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | Necesaria cuando `LLM_PROVIDER=anthropic` |

Para Anthropic, cambia `LLM_PROVIDER` a `anthropic` y ajusta `LLM_MODEL` al modelo que tengas disponible.

## Ejecución

```bash
python main.py
```

El script te pide una pregunta. También podés pasarla directamente:

```bash
python main.py "Explicame qué es la entropía"
```

La misma pregunta se envía una vez en modo normal y otra en streaming. Cada ejecución hace dos consultas a la API configurada.

Para ejecutar las pruebas locales sin consumir créditos de API:

```bash
python -m unittest discover -s tests -v
```

## Uso desde Python

```python
from pathlib import Path
from unified_llm import ChatMessage
from unified_llm.settings import manager_from_env

manager = manager_from_env(Path(".env"))
messages = [ChatMessage(role="user", content="Hola")]

response = await manager.generate(messages)
if response.error:
    print(response.error.code, response.error.message)
else:
    print(response.content)

async for event in manager.stream(messages):
    if event.error:
        print(event.error.code, event.error.message)
        break
    print(event.text, end="", flush=True)
```

`ModelConfig` valida temperatura de 0 a 2 (hasta 1 para Anthropic) y `max_tokens` mayor que cero. Los errores de autenticación, límite de tasa, conexión y API se devuelven en `ModelResponse.error` o en un `StreamEvent.error`; `retryable` indica si conviene reintentar.

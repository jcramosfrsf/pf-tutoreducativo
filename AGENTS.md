# AGENTS.md — Tutor Educativo

**Tutor educativo**: Sistema que responde preguntas sobre material de un curso, genera ejercicios y evalúa respuestas.

Instruction material for AI coding agents (GitHub Copilot, Claude, etc.) working on this repository.

---

# Características del sistema
El sistema se alimenta con 'textos educativos' de un curso (apuntes, libros, ejercicios, respuestas de ejercicios, etc.) y debe ser capaz de:
- Responder con un resumen de un concepto del curso
- Generar un cuestionario de múltiple opción, o de desarrollo de respuestas, y luego sea capaz de evaluar las respuestas al cuestionario realizadas por un alumno.

---

## Repository layout

```
core/
  - config.py <-- Centralized environment-based configuration.
  - llm_client.py  <-- Provider-agnostic LLM client abstraction (Gemini & Groq via OpenAI API).
  - logger.py  <-- Application logging setup.
  - tokenlab.py <-- Token counting, cost estimation, latency measurement, and budget checking. Supports Gemini and Groq providers

chroma_db/ ← Persisted ChromaDB vector store (generated)

data/   <-- Source of data

main_rag.py <--  RAG management

prompting/ <-- Prompt engineering toolkit: templates, chains, registry, and evaluation
    templates/ <-- prompt templates 


rag/  <-- RAG process

```

---

## Tool contract (never break these signatures)

| Tool | Python signature | Returns |

All tools are synchronous and return JSON-serialisable values. Do not add `async` to tool functions without updating the FastMCP configuration.

---

## Technology choices

| Concern | Choice | Rationale |
|---|---|---|
| MCP server | `fastmcp` | Pythonic, minimal boilerplate |
| Vector store | `chromadb` (persistent) | Local, no external service |
| Embeddings | `chromadb` `DefaultEmbeddingFunction` (`all-MiniLM-L6-v2` via ONNXRuntime) | No PyTorch / CUDA required |
| Relational DB | SQLite via `sqlalchemy` | Zero-config, file-based |
| PDF parsing | `pdfplumber` | Handles WHO PDF multi-column layout |
| Data transforms | `polars` | All ingestion transforms use Polars DataFrames |
| Package manager | `uv` | Fast, reproducible, single lock file |

---

## Coding conventions

- Python ≥ 3.11. Use `from __future__ import annotations` in every module.
- All data transformations in `embeddings.py` and `ingestion.py` and array/table operations must use **Polars** — never pandas.
- Format with `ruff format`, lint with `ruff check --fix`.
- No secrets in source — all paths come from environment variables loaded  via `python-dotenv` (see `.env.example`).
- SQL Querys must reject any non-SELECT statement before touching the database.
- Always use `upsert_interaction`, never raw INSERT in new code.

---

## Development commands

```bash
# Install all dependencies (creates .venv automatically)
uv sync

# Run the MCP server — stdio transport (default for MCP clients)

# Run with FastMCP dev inspector — browser UI on http://localhost:5173
uv run fastmcp dev <server>


# Lint + format
uv run ruff check --fix src/
uv run ruff format src/

# Tests
uv run pytest
```

---

## Agent reasoning flow

---

## Extending the system

### Adding a new tool
1. Define it as a `@mcp.tool()` decorated function in <...>.
2. Add its signature to the Tool contract table above.
3. Write a unit test in `tests/`.


For **GitHub Copilot in VS Code**, place this in `.vscode/mcp.json`.

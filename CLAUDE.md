# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management (Python 3.13+, no separate lockfile-less installs). **Always use `uv` (e.g. `uv run ...`, `uv sync`, `uv add`) — do not use `pip` directly.**

```bash
# Install dependencies
uv sync

# Run the app (from repo root)
chmod +x run.sh && ./run.sh

# Run manually with auto-reload
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app serves the frontend at `http://localhost:8000` and the API docs at `http://localhost:8000/docs`. There is no build step, linter, or test suite configured in this repo.

Requires a `.env` file in the repo root with `ANTHROPIC_API_KEY=...` (see `.env.example`).

## Architecture

This is a Retrieval-Augmented Generation (RAG) chatbot for course materials: FastAPI backend, ChromaDB vector store, Anthropic Claude for generation, and a static HTML/JS/CSS frontend (no build tooling, no framework).

### Request flow

`frontend/script.js` → `POST /api/query` (`backend/app.py`) → `RAGSystem.query()` (`backend/rag_system.py`) → `AIGenerator.generate_response()` (`backend/ai_generator.py`), which calls Claude with the `search_course_content` tool available.

Claude decides whether to answer directly (general knowledge) or invoke the tool (course-specific questions). If it invokes the tool, `AIGenerator._handle_tool_execution()` runs the tool, feeds the result back to Claude in a second (tools-disabled) call, and returns that final synthesis. **At most one tool call round-trip happens per query** — the system prompt in `AIGenerator.SYSTEM_PROMPT` enforces "one search per query maximum."

### Core components (backend/)

- **`rag_system.py`** — Central orchestrator. Wires together document processing, vector storage, AI generation, sessions, and tools. `query()` is the main entry point.
- **`ai_generator.py`** — Wraps the Anthropic client, owns the system prompt, and handles the tool-use round trip.
- **`search_tools.py`** — `Tool` ABC + `CourseSearchTool` (semantic search with optional `course_name`/`lesson_number` filters) + `ToolManager` (registers tools, tracks last-used sources for UI display, gives Claude the tool definitions).
- **`vector_store.py`** — ChromaDB wrapper with two collections: `course_catalog` (course titles/instructors/lesson metadata, used to fuzzy-resolve partial course names via vector search) and `course_content` (chunked text for semantic search). `search()` is the unified interface: resolves course name → builds filter → queries `course_content`.
- **`document_processor.py`** — Parses course transcript `.txt` files from `docs/` into `Course`/`Lesson`/`CourseChunk`. Expects a specific header format (`Course Title:`, `Course Link:`, `Course Instructor:` on the first lines) followed by `Lesson N: Title` markers. Chunking is custom sentence-boundary regex (handles abbreviations), not a library, with configurable size/overlap by character count.
- **`models.py`** — Pydantic models: `Course`, `Lesson`, `CourseChunk`.
- **`session_manager.py`** — In-memory (non-persistent, resets on restart) conversation history keyed by session ID, capped at `MAX_HISTORY` exchanges.
- **`config.py`** — Dataclass config loaded from `.env`: `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `EMBEDDING_MODEL` (`all-MiniLM-L6-v2`), `CHUNK_SIZE`/`CHUNK_OVERLAP`, `MAX_RESULTS`, `MAX_HISTORY`, `CHROMA_PATH`.
- **`app.py`** — FastAPI entrypoint. `POST /api/query` and `GET /api/courses`, serves `frontend/` as static files, and on startup loads any `.txt`/`.pdf`/`.docx` course documents from `../docs` into the vector store (skips courses already indexed by title).

### Data model notes

- Course `title` is used as the unique ID/primary key throughout (vector store IDs, catalog lookups) — there is no separate course ID field.
- Course name matching in searches is fuzzy (vector search against the `course_catalog` collection), not exact string match — `CourseSearchTool` accepts partial names like `"MCP"` or `"Introduction"`.
- Adding new course documents: drop `.txt` files into `docs/` following the header format described above; they're picked up automatically on next server startup (existing course titles are skipped, not re-processed).

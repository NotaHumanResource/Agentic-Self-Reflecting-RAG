# QWEN Installation Guide

End-to-end instructions for installing, configuring, and running the QWEN
agentic RAG system on Windows. This document targets a competent developer
reproducing the reference environment for the first time. It does not assume
prior knowledge of Ollama, Qdrant, or the project's internal architecture.

For a conceptual overview of what the system does, see
`autonomous_cognition_overview.md` in this folder.

---

## What QWEN Is

QWEN is a locally-hosted autonomous AI assistant built around a quantized
27B-parameter language model running on Ollama. It combines:

- A Streamlit web interface for chat, voice, image, and document interaction
- A dual-database memory system (SQLite for relational data, Qdrant for vector
  search)
- A bracketed command protocol (`[STORE:]`, `[SEARCH:]`, `[REMINDER:]`, etc.)
  that lets the model write and retrieve memories autonomously
- Background cognitive tasks (reflection, consolidation, drift detection,
  self-initiated wandering) running on independent schedules
- An OODA deep-research loop for multi-turn web inquiry
- Optional speech (Whisper STT, Kokoro TTS) and vision (Qwen3-VL) modalities
- An authenticated multi-user setup

Everything runs locally. The only outbound network calls are optional: the
DISCUSS_WITH_CLAUDE feature (Anthropic API) and web search for the OODA loop
and knowledge-gap filling.

---

## Hardware Requirements

### Reference configuration (developer machine)

- **GPU:** NVIDIA RTX 5090, 32 GB VRAM
- **CPU:** Intel i9, 24 cores
- **RAM:** 64 GB DDR5
- **OS:** Windows 11

### Minimum viable configuration

The system will run with reduced performance on lesser hardware, but the
defaults in `config.py` assume the reference setup. Below the following
thresholds, you will need to tune `MODEL_PARAMS`, `HARDWARE_CONFIG`, and
`OLLAMA_ENV_CONFIG` (see Configuration section):

- **GPU:** NVIDIA with at least 24 GB VRAM for the dense 27B model with 65K
  context. Smaller VRAM forces either a smaller model (e.g., a 7B variant) or
  a reduced `num_ctx` value.
- **CPU:** 8+ cores recommended (the cognitive loop, autonomous tasks, and
  Streamlit run on separate threads).
- **RAM:** 32 GB minimum. 64 GB is needed to comfortably keep the KV cache,
  document chunks, and Qdrant payloads resident.
- **Disk:** At least 200 GB free. Ollama model files alone are 20–60 GB each,
  and you'll likely want three: chat, embedding, and vision.
- **OS:** Windows 10/11. The codebase contains Windows-specific paths and
  Win32 file-locking workarounds. Running on Linux/macOS will require
  modifications.

### Network

A working internet connection is required only during installation (model
pulls, package installs) and for optional features (web search, Claude API).
Day-to-day operation is fully offline.

---

## Software Prerequisites

Install these in the order listed before proceeding to the project setup.

### 1. Python 3.11+

The codebase uses features and packages requiring Python 3.11 or later
(`pydantic 2.x`, `numpy 2.x`, `ipython 9.x`). Python 3.13 is the version on
the reference machine.

Download from https://www.python.org/downloads/. During installation:

- Check **Add Python to PATH**
- Use the default installer location or place under your user directory

Verify:

```
python --version
pip --version
```

### 2. Git for Windows

Required for cloning the repository and for runtime `GitPython` dependency.

Download from https://git-scm.com/download/win and accept the defaults.

Verify:

```
git --version
```

### 3. Ollama for Windows

Ollama hosts the local LLM. The reference machine runs the standard Windows
installer.

Download from https://ollama.com/download/windows. After install, Ollama lives
at `C:\Users\<your-username>\AppData\Local\Programs\Ollama\ollama.exe`.

Verify (in a new terminal so PATH refreshes):

```
ollama --version
```

### 4. Docker Desktop for Windows

Qdrant (the vector database) runs as a Docker container. Although Qdrant can
also be embedded locally as a Python library, the project is configured for
the Docker-hosted server mode (`QDRANT_USE_LOCAL = False` in `config.py`).

Download from https://www.docker.com/products/docker-desktop. Docker Desktop
requires WSL 2 or Hyper-V. Accept the WSL 2 backend when prompted — it has
fewer compatibility issues than Hyper-V.

After installation, launch Docker Desktop and let it finish first-run setup.
The Docker engine must be running before you start Qdrant.

Verify:

```
docker --version
docker run --rm hello-world
```

### 5. Optional: NVIDIA CUDA Toolkit

Ollama bundles its own CUDA runtime, but installing the full CUDA Toolkit
gives you `nvidia-smi` for monitoring GPU usage during runtime. Match the
version Ollama expects (currently CUDA 12.x).

Download from https://developer.nvidia.com/cuda-downloads.

Verify:

```
nvidia-smi
```

---

## Installation

### Step 1 — Clone the repository

```
cd C:\Users\<your-username>\source\repos
git clone <repository-url> Ollama3
cd Ollama3
```

The default path that `config.py` expects is
`C:\Users\<your-username>\source\repos\Ollama3`. You can clone elsewhere, but
note that one file currently hardcodes this path (see Known Gotchas).

### Step 2 — Create and activate a Python virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` in your prompt. Activate this environment in any
terminal that runs the project or its install scripts.

### Step 3 — Install Python dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
pip install concurrent-log-handler
```

The second `pip install` is intentional: `concurrent_log_handler` is imported
in `config.py` but is not currently listed in `requirements.txt`. The system
will refuse to import without it. See Known Gotchas.

Installation will take 10–20 minutes. Some packages (`tensorflow`-adjacent,
`PyQt6`, `Panda3D`, `vpython`) pull large binaries.

If you do not need optional features (speech, vision, visualization), you can
selectively skip those packages. Required core packages include `streamlit`,
`langchain-core`, `langchain-ollama`, `langchain-qdrant`, `qdrant-client`,
`ollama`, `anthropic`, `bcrypt`, `streamlit-authenticator`, `pydantic`,
`numpy`, `pandas`, `requests`.

### Step 4 — Pull the Ollama models

Start Ollama in a separate terminal:

```
ollama serve
```

In a different terminal, pull the required models:

```
ollama pull huihui_ai/Qwen3.6-abliterated:27b
ollama pull qwen3-embedding:8b
```

Optional for vision and the alternative chat model:

```
ollama pull qwen3-vl:30b
ollama pull huihui_ai/qwen3-coder-abliterated:30b-a3b-instruct-q3_K_M
```

Each model is 15–60 GB. The pulls can take significant time on a slow
connection.

### Step 5 — Build the slim embedding model

`config.py` references a custom embedding model `qwen3-embedding:slim`. This
is a derivative of the standard `qwen3-embedding:8b` rebuilt with
`num_ctx=2048` instead of the upstream default 40960, to allow dual-model
resident loading on a 32 GB GPU.

Create a file named `Modelfile.embedding-slim` containing:

```
FROM qwen3-embedding:8b
PARAMETER num_ctx 2048
```

Then build:

```
ollama create qwen3-embedding:slim -f Modelfile.embedding-slim
```

Verify all expected models are present:

```
ollama list
```

You should see at minimum `huihui_ai/Qwen3.6-abliterated:27b` and
`qwen3-embedding:slim`.

If your GPU has 48 GB or more of VRAM, you can skip the slim variant and edit
`config.py` to set `EMBEDDING_MODEL = "qwen3-embedding:8b"`.

### Step 6 — Start the Qdrant Docker container

In a new terminal (with Docker Desktop running):

```
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

This pulls the Qdrant image, creates a named volume for persistent storage,
and exposes ports 6333 (REST) and 6334 (gRPC). Confirm it's running:

```
docker ps
```

You should see a `qdrant/qdrant` container listed. Test it:

```
curl http://localhost:6333/
```

You should get a JSON response from Qdrant.

The Qdrant collections (`deepseek_memory_optimized`, `knowledge_gaps_embeddings`)
will be auto-created the first time the project starts. No manual schema
setup is required.

### Step 7 — Configure authentication

The Streamlit interface requires login. The reference `users.json` contains a
single admin user with a placeholder bcrypt hash. Generate your own:

```
python hash_password.py
```

Edit `hash_password.py` first to set the password you want hashed. Run the
script and copy the printed hash into `users.json`:

```json
{
  "credentials": {
    "usernames": {
      "yourname": {
        "email": "you@example.com",
        "name": "Your Name",
        "password": "<paste-bcrypt-hash-here>"
      }
    }
  },
  "cookie": {
    "name": "qwen_auth",
    "key": "<change-this-to-a-long-random-string>",
    "expiry_days": 7
  },
  "preauthorized": []
}
```

**Important:** the `cookie.key` is a session-signing secret. Change it from
the default before exposing the interface to anything beyond localhost.

### Step 8 — Optional: Configure Claude API key

If you want the `DISCUSS_WITH_CLAUDE` feature and the OODA loop's Claude
fallback, create a file named `ClaudeAPIKey.txt` in the project root
containing your Anthropic API key on a single line. Get a key from
https://console.anthropic.com/. The file is read once at startup; nothing
else needs configuration.

If this file is missing, `DISCUSS_WITH_CLAUDE` will fail silently and the
OODA loop will fall back to direct web search.

---

## Configuration

The project has multiple configuration surfaces. Most users only need to
touch `config.py` and `system_config.json`.

### `config.py` — Hardware and model defaults

This file is the central configuration. It is tuned for the reference
hardware. Key sections to review:

**Paths and database (lines 122–135):**

- `BASE_DIR` resolves to the project root automatically.
- `DOCS_PATH` (`LocalDocs/`) is where document uploads are stored.
- `DB_PATH` (`LongTermMemory_data.db`) is the SQLite database file. It is
  auto-created on first run.
- `QDRANT_URL = "http://localhost:6333"` — leave unchanged unless you run
  Qdrant on a different host.
- `QDRANT_USE_LOCAL = False` — keep `False` to use the Docker container. Set
  `True` and uncomment the `QDRANT_LOCAL_PATH` block if you want embedded
  Qdrant instead, at the cost of significant performance loss.

**Model names (lines ~165–195):**

- `OLLAMA_MODEL` — primary chat model. Currently set to a test variant; for
  production use, set this to `"huihui_ai/Qwen3.6-abliterated:27b"`.
- `EMBEDDING_MODEL = "qwen3-embedding:slim"` — the custom-built embedding
  model from Step 5. Change to `"qwen3-embedding:8b"` if you skipped that
  step.
- `IMAGE_MODEL` / `VIDEO_MODEL` — used by the image and video analysis
  features. Defaults to the primary chat model (which is multimodal). Set to
  `"qwen3-vl:30b"` if you pulled the vision model and want it routed there.

**Model parameters (`MODEL_PARAMS`, lines ~200–250):**

- `num_ctx`: 65536 tokens. Reduce to 32768 or 16384 if you have less VRAM.
- `num_predict`: 4608 tokens max per response.
- `temperature`, `top_k`, `top_p`: sampling controls; leave at defaults
  unless you have a specific need.
- `presence_penalty`: 0.3 (deliberately low for command-syntax reliability;
  see comment in file).
- `num_thread`: 24. Set to your CPU's physical core count.
- `num_gpu_layers`: -1 (all on GPU). Set to a positive integer if your VRAM
  cannot hold the full model.

**Hardware config (`HARDWARE_CONFIG`, lines ~270–320):**

- `gpu_memory_fraction`: 0.95 (use 95% of VRAM). Lower to 0.85 if you run
  other GPU workloads concurrently.
- `cpu_threads`: 24. Match your core count.
- `memory_pool_size`: `"52GB"`. Lower if you have less than 64 GB RAM.
- `kv_cache_size`: `"16GB"`. Affects how much context the model can keep
  active. Reduce proportionally if you reduced `num_ctx`.

**Ollama environment (`OLLAMA_ENV_CONFIG`, lines ~330–380):**

These map to environment variables Ollama reads. Most are duplicated in
`Startup Scripts/Ollama.bat`. Critical values:

- `OLLAMA_KEEP_ALIVE = "24h"` — keeps models loaded so chat doesn't pay
  reload cost.
- `OLLAMA_MAX_LOADED_MODELS = "2"` — allows the chat model and embedding
  model to coexist in VRAM.
- `OLLAMA_FLASH_ATTENTION = "1"` — required for 65K context on 32 GB VRAM.
- `OLLAMA_CONTEXT_SIZE = "65536"` — must match `num_ctx`.

### `system_config.json` — Autonomous cognition toggles

```json
{
  "autonomous_thinking_disabled": true,
  "memory_management_disabled": true,
  "disabled_cognitive_activities": [ ... ]
}
```

The reference repo ships with all autonomous activities **disabled** for
safety on a fresh install. You can enable them individually from the
Streamlit UI sidebar (Autonomous Thinking expander) after the system is
running and you've verified everything connects properly.

To enable autonomous cognition on first launch, edit the file so both top
booleans are `false` and remove activities you want to enable from
`disabled_cognitive_activities`. The full list of activity names is
documented in `autonomous_cognition_overview.md`.

### `reflection_config.json` — Reflection schedule toggles

```json
{"daily": false, "weekly": true, "monthly": true}
```

Per-tier toggles for the daily / weekly / monthly reflection schedule.
Editable from the UI sidebar (Self-Reflection expander) once the system is
running. Reflection times are hardcoded in `autonomous_cognition.py` at
06:15 daily, Sunday 09:15 weekly, and 1st of month 12:20 monthly.

### Optional configuration files (auto-created)

- `speech_settings.json` — STT/TTS preferences. Created on first save from
  the UI; no manual setup needed.
- `failed_domains.json`, `search_blocked_domains.json` — web search
  blocklists. Empty defaults are fine.

---

## Directory Structure

Most directories are auto-created on first launch. You should not need to
create any of them manually. For reference, here's what each one holds:

| Path | Purpose | Auto-created |
| --- | --- | --- |
| `LocalDocs/` | Document uploads (PDF, DOCX, TXT, MD, RTF) | Yes |
| `reflections/` | JSON completion files for scheduled reflections; thought files from autonomous tasks | Yes |
| `Ollama_logs/` | Rotating application logs (`ollama_context.log` and rotations) | Yes |
| `image_uploads/` | Image files uploaded for vision analysis | Yes (on first upload) |
| `video_uploads/` | Video files uploaded for analysis | Yes (on first upload) |
| `search_logs/` | OODA loop and web search debug logs | Yes |
| `kokoro_models/` | Kokoro TTS voice model files | Manually if using TTS |
| `Startup Scripts/` | Batch files for launching the system | Part of repo |
| `guides/` | Generated command reference HTML | Yes |
| `Documentation/` | This guide and related docs | Manual |

**Files created at the project root by the running system:**

- `LongTermMemory_data.db` — primary SQLite database (memories, reminders,
  cognitive state history, lifetime counters)
- `LongTermMemory_data.db-shm`, `LongTermMemory_data.db-wal` — SQLite WAL
  journal files (auto-managed)
- `LifetimeCounters.db` — separate SQLite database for cross-session command
  counters
- `scheduler.lock` — lock file preventing duplicate scheduler instances
- `AI.log` — fallback log file
- `sync_reports.json`, `sync_results.json` — database maintenance output

### Database notes

**SQLite** requires no separate installation. The schema is created
automatically on first connection (`_upgrade_db_schema()` in
`memory_db.py`). The default database file is `LongTermMemory_data.db` in
the project root.

The schema includes:

- `memories` table — primary memory store, all types
- `reminders` table — date-keyed reminders
- `deletion_queue` table — pending Qdrant deletions for transaction
  coordination
- (and others created on demand by various subsystems)

**Qdrant** also requires no manual setup. Collections are auto-created on
first connection with the vector size (4096), distance metric (cosine), HNSW
parameters, and quantization config defined in `config.py` `QDRANT_CONFIG`.

To name the databases differently, edit `DB_PATH` and
`QDRANT_COLLECTION_NAME` in `config.py`. Changing these on an existing
install will leave the old data orphaned; migrate or start fresh.

---

## First Launch

Once installation and configuration are complete, the launch sequence is:

### 1. Start Docker Desktop and verify Qdrant is running

```
docker ps
```

If the `qdrant` container is not listed, start it:

```
docker start qdrant
```

### 2. Start Ollama

In a dedicated terminal:

```
ollama serve
```

Leave this terminal running. Alternatively, run
`Startup Scripts\Ollama.bat`, which sets the environment variables and
launches the server in the background.

Verify the model loads correctly:

```
curl http://localhost:11434/api/tags
```

### 3. Activate the Python venv and launch Streamlit

In a new terminal in the project root:

```
.venv\Scripts\activate
streamlit run main.py
```

Streamlit will open a browser tab at `http://localhost:8501`.

Alternatively, run `Startup Scripts\QWEN.bat` which performs the cleanup,
verification, and Streamlit launch in one step. Note that QWEN.bat does
**not** start Ollama — that must be started separately first.

### 4. Log in

The login screen will appear. Use the username and password you set in
`users.json` during Step 7 of installation. Successful login lands you on
the chat interface.

### 5. Verify the system is healthy

In the chat input, send a simple greeting. You should see the model
respond. Then verify the major subsystems via the sidebar:

- **System Maintenance → Run Enhanced Health Check** — confirms both
  databases are reachable and synchronized.
- **Self-Reflection → Run Reflection Now → Daily** — confirms the
  reflection engine and LLM work end-to-end.
- **File Import → upload a small PDF** — confirms document processing and
  vector embedding work.

If any of these fail, check `Ollama_logs/ollama_context.log` for
diagnostics.

---

## Known Gotchas and Deficiencies

Items here are confirmed install-time issues that a fresh developer is
likely to hit. They should be addressed when convenient but are not
blockers if you work around them.

**`concurrent_log_handler` is missing from `requirements.txt`.** This
package is imported at the top of `config.py` and the system cannot start
without it. Install separately with `pip install concurrent-log-handler`
after the main `pip install -r requirements.txt`. The right long-term fix
is to add it to `requirements.txt`.

**Hardcoded user path in `autonomous_cognition.py`.** Line ~1760 contains
`REFLECTIONS_PATH = r"C:\Users\kenba\source\repos\Ollama3\reflections"`,
which will not resolve on any machine where the repo is at a different
path. The autonomous audit-memory-confidence task writes its report
there. If you clone to a different location, either edit this line to
match your path or use `os.path.join(BASE_DIR, "reflections")` as is done
elsewhere in the file.

**`OLLAMA_MODEL` in `config.py` does not match `CLAUDE.md`.** The
reference `CLAUDE.md` documents the primary model as
`huihui_ai/Qwen3.6-abliterated:27b`, but the current `config.py` has
`OLLAMA_MODEL` set to a different test model. Verify the value before
your first launch and set it to the model you actually want as primary.

**Custom embedding model must be built locally.** Pulling
`qwen3-embedding:8b` from Ollama is not enough; the system expects
`qwen3-embedding:slim`. See Step 5. Alternatively edit `EMBEDDING_MODEL`
in `config.py` to point at the upstream variant.

**Hardware defaults assume RTX 5090 / i9 / 64 GB.** Values in
`MODEL_PARAMS`, `HARDWARE_CONFIG`, and `OLLAMA_ENV_CONFIG` are tuned for
this setup. On lesser hardware, the system may run but with severe
performance penalties or OOM errors. See the Configuration section for
which values to scale down.

**Default `users.json` cookie key is a placeholder.** The shipped value
in the `cookie.key` field is a development placeholder
(`"sk_synthetic_ai_2025_change_this_secret_key"`). Change it to a
long random string before any non-localhost deployment.

**Windows-only.** `config.py` contains Win32-specific console encoding
fixes and `ConcurrentRotatingFileHandler` to work around Windows file
locking. Several other modules contain `taskkill` and other Win32-only
calls. A Linux or macOS port is feasible but requires removing or
replacing these portions.

**`Main.py` vs `main.py` case difference.** `Startup Scripts/QWEN.bat`
runs `streamlit run Main.py` with a capital `M`. The actual file is
lowercase `main.py`. This works on Windows due to case-insensitive
file lookups but would break verbatim on Linux.

**Streamlit autorefresh is optional but warned about.** If
`streamlit-autorefresh` is missing, wake-word polling is disabled but
the rest of the system runs. The startup log will show a clear warning
in this case.

**Speech features are optional and isolated.** `whisper_speech.py`,
`speech_utils.py`, and `wake_word_listener.py` are wrapped in
try/except imports. If any fail, the corresponding feature flag is
disabled and the rest of the system continues. You do not need
PyAudio, Whisper, or Kokoro models unless you want voice interaction.

---

## Quick Reference: Minimum Files to Edit Before First Launch

If you want the shortest possible setup checklist:

1. Clone the repo and `pip install -r requirements.txt` and
   `pip install concurrent-log-handler`.
2. Pull `huihui_ai/Qwen3.6-abliterated:27b` and `qwen3-embedding:8b` via
   Ollama.
3. Build `qwen3-embedding:slim` via the Modelfile in Step 5 (or skip and
   edit `config.py` to use the upstream `:8b` model).
4. Start Docker, run the Qdrant container per Step 6.
5. Generate a bcrypt hash with `hash_password.py` and place it in
   `users.json`. Change the cookie key.
6. In `config.py`, confirm `OLLAMA_MODEL` is set to
   `"huihui_ai/Qwen3.6-abliterated:27b"`.
7. Fix the hardcoded reflections path on line ~1760 of
   `autonomous_cognition.py` if you cloned to a non-default location.
8. Start Ollama, run `streamlit run main.py`, log in.

That should be enough to reach a working first chat. Everything else can
be tuned from the UI sidebar after the system is live.

---

*Last updated: 2026-05-26.*

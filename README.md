---
title: Bugs Buddy Environment Server
emoji: 🐛
colorFrom: purple
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - software-engineering
  - debugging
---

# Bugs Buddy

**RL environment for autonomous bug root cause analysis.**

An agent receives a bug report and access to a small Python codebase. It must
take a sequence of investigative actions — reading files, running tests,
searching code — to identify the precise root cause and submit a hypothesis
for grading.

Built for the Meta × PyTorch OpenEnv Hackathon.

---

## Environment

### Action space

`BugsBuddyAction` — one of 5 investigative tools:

| Tool | Args | Description |
|---|---|---|
| `read_file` | `{filename}` | Read a source file by name |
| `search_code` | `{query}` | Search all files for a string pattern |
| `run_tests` | `{}` | Get pre-computed test output |
| `inspect_function` | `{function_name}` | Get full source of a named function |
| `submit_root_cause` | `{filename, line, explanation}` | Submit hypothesis (terminal) |

### Observation space

`BugsBuddyObservation` — returned after every step:

- `tool_output` — result of the last tool call
- `bug_report` — title, description, stack trace, task ID, difficulty
- `available_files` — list of all filenames in the codebase
- `steps_remaining` — steps left before timeout
- `action_history` — summary of all prior actions this episode

### Reward

| Event | Reward |
|---|---|
| First `run_tests` | +0.05 |
| First read of each relevant file (max 2) | +0.05 each |
| `inspect_function` on the buggy function | +0.10 |
| Search that hits the buggy file | +0.05 |
| Invalid / repeated action | −0.05 |
| Timeout (20 steps, no submission) | −0.10 |
| Terminal: `grader_score × max(0.5, 1 − steps/20 × 0.5)` | 0.0–1.0 |

---

## Tasks

### task_easy — Discount Calculator (1 file, ~2–4 steps)

`get_final_price()` adds the discount amount instead of subtracting it.
Test output shows three failing assertions. Expected agent steps: 2–4.

### task_medium — Operator Precedence (3 files, ~6–12 steps)

`compare_groups()` computes `mean_a / mean_b * 100`, which Python evaluates
as `mean_a / (mean_b * 100)` — 100× too small. All tests pass; no test
covers `compare_groups()` directly. Expected agent steps: 6–12.

### task_hard — Pagination Off-by-One (4 files, ~10–18 steps)

`get_page()` boundary guard uses the batch size instead of total record count,
causing the last page to silently return `[]`. Requires tracing a data-flow
bug across `paginator.py` and `data_loader.py`. Expected agent steps: 10–18.

---

## Quick Start

```python
import asyncio
from bugs_buddy import BugsBuddyAction, BugsBuddyEnv, ToolName

async def main():
    async with BugsBuddyEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_id="task_easy")
        obs = result.observation
        print(obs.bug_report.title)
        print("Files:", obs.available_files)

        result = await env.step(BugsBuddyAction(
            tool=ToolName.run_tests, args={}
        ))
        print(result.observation.tool_output)

asyncio.run(main())
```

---

## Running Locally

```bash
# Install dependencies
uv sync

# Start the server
uv run uvicorn bugs_buddy.server.app:app --host 0.0.0.0 --port 8000

# Run the baseline inference script (requires API credentials)
ENV_BASE_URL=http://localhost:8000 uv run python inference.py
```

---

## Project Structure

```
bugs_buddy/
├── Dockerfile                  ← Root-level Docker build (used by HF Spaces)
├── openenv.yaml                ← OpenEnv manifest
├── pyproject.toml
├── uv.lock
├── inference.py                ← Baseline inference script
├── models.py                   ← BugsBuddyAction, BugsBuddyObservation, BugsBuddyState
├── client.py                   ← BugsBuddyEnv WebSocket client
└── server/
    ├── environment.py          ← Core RL environment (reset, step, tool handlers)
    ├── graders.py              ← Deterministic scorers (easy / medium / hard)
    ├── app.py                  ← FastAPI app
    └── tasks/
        ├── task_easy.py        ← Discount calculator bug
        ├── task_medium.py      ← Operator precedence bug
        └── task_hard.py        ← Pagination off-by-one bug
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode (body: `{task_id?: string}`) |
| `/step` | POST | Execute one action |
| `/state` | GET | Current episode metadata |
| `/schema` | GET | Action / observation JSON schemas |
| `/health` | GET | Health check |
| `/ws` | WebSocket | Persistent session (low-latency multi-step) |
| `/web` | GET | Interactive web UI |
| `/docs` | GET | Swagger / OpenAPI docs |

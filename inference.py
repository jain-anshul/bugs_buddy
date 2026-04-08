"""
Bugs Buddy — Baseline Inference Script
=======================================
Hackathon baseline for the Bugs Buddy RL environment.

MANDATORY environment variables:
    API_BASE_URL   LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key / Hugging Face token

STDOUT FORMAT (one line per event):
    [START] task=<task_id> env=bugs_buddy model=<model>
    [STEP]  step=<n> action=<tool>(<key_arg>) reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

The script runs all three tasks sequentially and writes results.json to the
project root when finished.
"""

import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional, Dict, Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME", "")

MAX_STEPS = 20
TEMPERATURE = 0.3
MAX_TOKENS = 1024
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["task_easy", "task_medium", "task_hard"]
BENCHMARK = "bugs_buddy"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a software debugging agent. Your job is to identify the root cause
    of a bug in a Python codebase.

    You have the following tools:

      read_file        — Read a source file by name.
                         Args: {"filename": "path/to/file.py"}

      search_code      — Search all files for a string pattern.
                         Args: {"query": "some code pattern"}

      run_tests        — Get the pre-computed test output for this task.
                         Args: {}

      inspect_function — Get the full source of a named function.
                         Args: {"function_name": "my_func"}

      submit_root_cause — Submit your root cause hypothesis (terminal action).
                          Args: {"filename": "path/to/file.py",
                                 "line": "42",
                                 "explanation": "...detailed explanation..."}

    At each step, reason briefly about what you know, then emit exactly one
    ACTION line in this format:

        ACTION: {"tool": "<tool_name>", "args": {<args>}}

    When you are confident about the root cause, use submit_root_cause.
    Be specific in your explanation — mention the operator, variable, or logic
    error, the function name, and why it causes the observed failure.

    To find the exact line number: use read_file on the relevant file and
    locate the specific buggy line before submitting. inspect_function does
    not return line numbers.

    You have a limited step budget. Investigate efficiently.
""").strip()


def build_user_message(obs_data: Dict[str, Any], step: int) -> str:
    """Build the user message for the current step."""
    parts = []

    if step == 1 and obs_data.get("bug_report"):
        br = obs_data["bug_report"]
        parts.append(f"BUG REPORT\nTitle: {br.get('title', '')}\n{br.get('description', '')}")
        if br.get("stack_trace"):
            parts.append(f"Stack trace:\n{br['stack_trace']}")
        files = obs_data.get("available_files", [])
        parts.append(f"Available files: {', '.join(files)}")
        parts.append(f"Steps remaining: {obs_data.get('steps_remaining', MAX_STEPS)}")
    else:
        parts.append(f"Step {step} | Steps remaining: {obs_data.get('steps_remaining', 0)}")
        output = obs_data.get("tool_output", "")
        parts.append(f"Last tool output:\n{output}")

    return "\n\n".join(parts)


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract the ACTION JSON from the model response."""
    match = re.search(r"ACTION:\s*(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def get_llm_action(client: OpenAI, messages: List[Dict]) -> tuple[Optional[Dict[str, Any]], str]:
    """Call the LLM and parse its tool action."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_action(text), text
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return None, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

    try:
        from bugs_buddy.client import BugsBuddyEnv
        from bugs_buddy.models import BugsBuddyAction, ToolName
    except ImportError:
        from client import BugsBuddyEnv
        from models import BugsBuddyAction, ToolName

    server_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")

    all_results = []

    for task_id in TASKS:
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False
        grader_score = 0.0

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        if IMAGE_NAME:
            env = await BugsBuddyEnv.from_docker_image(IMAGE_NAME)
        else:
            env = BugsBuddyEnv(base_url=server_url)

        try:
            result = await env.reset(task_id=task_id)
            obs_dict = result.observation.model_dump() if hasattr(result.observation, "model_dump") else result.observation.__dict__

            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(obs_dict, step=1)},
            ]

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action_parsed, llm_text = get_llm_action(client, conversation)

                error_msg = None
                if action_parsed is None:
                    if step == 1:
                        action_parsed = {"tool": "run_tests", "args": {}}
                    elif step == 2 and obs_dict.get("available_files"):
                        action_parsed = {"tool": "read_file", "args": {"filename": obs_dict["available_files"][0]}}
                    else:
                        error_msg = "Could not parse action from model response"
                        action_parsed = {
                            "tool": "submit_root_cause",
                            "args": {
                                "filename": obs_dict.get("available_files", ["unknown.py"])[0],
                                "line": "1",
                                "explanation": "Unable to determine root cause from available information.",
                            },
                        }

                tool = action_parsed.get("tool", "unknown")
                args = action_parsed.get("args", {})
                key_arg = next(iter(args.values()), "") if args else ""
                action_str = f"{tool}({key_arg[:40]})" if key_arg else f"{tool}()"

                try:
                    action = BugsBuddyAction(
                        tool=ToolName(tool),
                        args={k: str(v) for k, v in args.items()},
                    )
                except Exception as e:
                    error_msg = str(e)
                    action = BugsBuddyAction(tool=ToolName.run_tests, args={})

                result = await env.step(action)
                obs_dict = result.observation.model_dump() if hasattr(result.observation, "model_dump") else result.observation.__dict__

                reward = result.reward or 0.0
                done = result.done
                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

                conversation.append({"role": "assistant", "content": llm_text or ""})
                conversation.append({"role": "user", "content": build_user_message(obs_dict, step=step + 1)})

                if done:
                    m = re.search(r"Grader score:\s*([\d.]+)", obs_dict.get("tool_output", ""))
                    if m:
                        grader_score = float(m.group(1))
                    break

            score = min(max(sum(rewards), 0.0), 1.0)
            success = grader_score >= SUCCESS_SCORE_THRESHOLD

        finally:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

        all_results.append({
            "task_id": task_id,
            "total_steps": steps_taken,
            "total_reward": round(score, 4),
            "grader_score": round(grader_score, 4),
            "success": success,
        })

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Task':<15} {'Steps':>6} {'Grader':>8} {'Reward':>8} {'OK':>4}")
    print("-" * 60)
    for r in all_results:
        ok = "YES" if r["success"] else "no"
        print(
            f"{r['task_id']:<15} {r['total_steps']:>6} "
            f"{r['grader_score']:>8.4f} {r['total_reward']:>8.4f} {ok:>4}"
        )
    print("=" * 60)

    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    asyncio.run(main())

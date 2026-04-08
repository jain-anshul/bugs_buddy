"""
Bugs Buddy Environment — core RL environment implementation.

The agent receives a bug report and access to a bundled Python codebase.
It may call up to max_steps investigative tools before submitting a root
cause hypothesis via submit_root_cause, which triggers grading.
"""

import random
import re
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import BugsBuddyAction, BugsBuddyObservation, BugReport, BugsBuddyState, ToolName
    from .tasks import ALL_TASKS
    from .graders import GRADERS
except ImportError:
    from models import BugsBuddyAction, BugsBuddyObservation, BugReport, BugsBuddyState, ToolName
    from server.tasks import ALL_TASKS
    from server.graders import GRADERS

MAX_STEPS = 20


class BugsBuddyEnvironment(Environment):
    """
    RL environment for autonomous bug root cause analysis.

    An agent receives a bug report and a bundled codebase (dict of
    filename → source string). It calls investigative tools to identify
    the root cause, then submits a hypothesis via submit_root_cause.

    Supports concurrent sessions (each instance is fully self-contained).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._task = None
        self._state = BugsBuddyState(episode_id=str(uuid4()), step_count=0)
        # Step-reward tracking (one-time bonuses)
        self._rewarded_tests_run = False
        self._rewarded_files: set[str] = set()
        self._rewarded_inspect_function = False
        self._rewarded_search_hit = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> BugsBuddyObservation:
        """Reset the environment, optionally selecting a specific task."""
        if seed is not None:
            random.seed(seed)

        if task_id is not None:
            if task_id not in ALL_TASKS:
                raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(ALL_TASKS.keys())}")
            self._task = ALL_TASKS[task_id]
        else:
            self._task = random.choice(list(ALL_TASKS.values()))

        eid = episode_id or str(uuid4())
        self._state = BugsBuddyState(
            episode_id=eid,
            step_count=0,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            files_read=[],
            tests_run=False,
            current_hypothesis=None,
        )
        self._rewarded_tests_run = False
        self._rewarded_files = set()
        self._rewarded_inspect_function = False
        self._rewarded_search_hit = False

        bug_report = BugReport(
            title=self._task.bug_report_title,
            description=self._task.bug_report_description,
            stack_trace=self._task.bug_report_stack_trace,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
        )

        welcome = (
            f"Episode started. Task: {self._task.task_id} ({self._task.difficulty}).\n"
            f"You have {MAX_STEPS} steps to investigate and submit a root cause.\n"
            f"Available tools: read_file, search_code, run_tests, inspect_function, submit_root_cause."
        )

        return BugsBuddyObservation(
            done=False,
            reward=None,
            tool_output=welcome,
            tool_success=True,
            bug_report=bug_report,
            available_files=list(self._task.files.keys()),
            steps_remaining=MAX_STEPS,
            action_history=[],
        )

    def step(self, action: BugsBuddyAction, **kwargs) -> BugsBuddyObservation:  # type: ignore[override]
        """Execute one investigative step and return the observation."""
        if self._task is None:
            return self._error_obs("Environment not initialized. Call reset() first.", 0)

        self._state.step_count += 1
        steps_remaining = MAX_STEPS - self._state.step_count

        # Dispatch
        tool = action.tool
        args = action.args or {}
        reward = 0.0
        tool_success = True
        done = False

        try:
            if tool == ToolName.read_file:
                tool_output, step_reward = self._handle_read_file(args)
                reward += step_reward

            elif tool == ToolName.search_code:
                tool_output, step_reward = self._handle_search_code(args)
                reward += step_reward

            elif tool == ToolName.run_tests:
                tool_output, step_reward = self._handle_run_tests()
                reward += step_reward

            elif tool == ToolName.inspect_function:
                tool_output, step_reward = self._handle_inspect_function(args)
                reward += step_reward

            elif tool == ToolName.submit_root_cause:
                tool_output, step_reward, done = self._handle_submit_root_cause(args)
                reward += step_reward

            else:
                tool_output = f"Unknown tool: {tool}"
                tool_success = False
                reward -= 0.05

        except KeyError as e:
            tool_output = f"Missing required argument: {e}"
            tool_success = False
            reward -= 0.05

        # Timeout check
        if not done and self._state.step_count >= MAX_STEPS:
            done = True
            reward -= 0.10
            tool_output += "\n[TIMEOUT] Maximum steps reached. Episode ended."

        # Update action history
        summary = self._action_summary(tool, args)
        history = list(self._state.current_hypothesis and
                       [f"hyp: {self._state.current_hypothesis[:60]}"] or [])
        # Re-derive full history from state (we store it in state for simplicity)
        if not hasattr(self._state, "_action_history_list"):
            object.__setattr__(self._state, "_action_history_list", [])
        self._state._action_history_list.append(summary)
        action_history = list(self._state._action_history_list)

        return BugsBuddyObservation(
            done=done,
            reward=reward,
            tool_output=tool_output,
            tool_success=tool_success,
            bug_report=BugReport(
                title=self._task.bug_report_title,
                description=self._task.bug_report_description,
                stack_trace=self._task.bug_report_stack_trace,
                task_id=self._task.task_id,
                difficulty=self._task.difficulty,
            ),
            available_files=list(self._task.files.keys()),
            steps_remaining=max(0, steps_remaining),
            action_history=action_history,
        )

    @property
    def state(self) -> BugsBuddyState:
        return self._state

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _handle_read_file(self, args: dict) -> tuple[str, float]:
        filename = args.get("filename", "").strip()
        if not filename:
            return "Error: 'filename' argument is required for read_file.", -0.05

        if filename not in self._task.files:
            available = ", ".join(self._task.files.keys())
            return f"File not found: '{filename}'. Available files: {available}", 0.0

        # Repeat read penalty
        if filename in self._state.files_read:
            content = self._task.files[filename]
            return content, -0.05

        self._state.files_read.append(filename)

        reward = 0.0
        gt = self._task.ground_truth
        # First-read bonus for relevant files (up to 2)
        if filename in gt.relevant_files and len(self._rewarded_files) < 2:
            if filename not in self._rewarded_files:
                self._rewarded_files.add(filename)
                reward += 0.05

        return self._task.files[filename], reward

    def _handle_search_code(self, args: dict) -> tuple[str, float]:
        query = args.get("query", "").strip()
        if not query:
            return "Error: 'query' argument is required for search_code.", -0.05

        results = []
        for filename, source in self._task.files.items():
            for lineno, line in enumerate(source.splitlines(), 1):
                if query.lower() in line.lower():
                    results.append(f"{filename}:{lineno}: {line}")

        if not results:
            return f"No matches found for query: '{query}'", 0.0

        output = "\n".join(results)

        reward = 0.0
        gt = self._task.ground_truth
        # One-time bonus if search hits the buggy file
        if not self._rewarded_search_hit:
            if any(gt.filename in r for r in results):
                self._rewarded_search_hit = True
                reward += 0.05

        return output, reward

    def _handle_run_tests(self) -> tuple[str, float]:
        reward = 0.0
        if not self._rewarded_tests_run:
            self._rewarded_tests_run = True
            self._state.tests_run = True
            reward += 0.05
        else:
            reward -= 0.05  # penalty for re-running
        return self._task.test_output, reward

    def _handle_inspect_function(self, args: dict) -> tuple[str, float]:
        function_name = args.get("function_name", "").strip()
        if not function_name:
            return "Error: 'function_name' argument is required for inspect_function.", -0.05

        pattern = re.compile(
            r"(def\s+" + re.escape(function_name) + r"\s*\(.*?)"
            r"(?=\ndef\s|\nclass\s|\Z)",
            re.DOTALL,
        )

        for filename, source in self._task.files.items():
            match = pattern.search(source)
            if match:
                func_source = match.group(0).rstrip()
                reward = 0.0
                gt = self._task.ground_truth
                if (function_name == gt.buggy_function and not self._rewarded_inspect_function):
                    self._rewarded_inspect_function = True
                    reward += 0.10
                return f"# {filename}\n{func_source}", reward

        return f"Function '{function_name}' not found in any file.", 0.0

    def _handle_submit_root_cause(self, args: dict) -> tuple[str, float, bool]:
        filename = args.get("filename", "").strip()
        line_str = args.get("line", "").strip()
        explanation = args.get("explanation", "").strip()

        if not filename or not line_str or not explanation:
            return (
                "Error: submit_root_cause requires 'filename', 'line', and 'explanation'.",
                -0.05,
                False,
            )

        try:
            line = int(line_str)
        except ValueError:
            return f"Error: 'line' must be an integer, got '{line_str}'.", -0.05, False

        self._state.current_hypothesis = explanation

        gt = self._task.ground_truth
        grader = GRADERS[self._task.task_id]
        raw_score = grader(filename, line, explanation, gt)

        # Efficiency multiplier: max(0.5, 1.0 - (steps/max_steps) * 0.5)
        efficiency = max(0.5, 1.0 - (self._state.step_count / MAX_STEPS) * 0.5)
        terminal_reward = round(raw_score * efficiency, 4)

        feedback = (
            f"Root cause submitted.\n"
            f"  Filename:    {filename}\n"
            f"  Line:        {line}\n"
            f"  Explanation: {explanation[:200]}\n"
            f"\nGrader score: {raw_score:.4f}  (efficiency x{efficiency:.2f})\n"
            f"Terminal reward: {terminal_reward:.4f}"
        )

        return feedback, terminal_reward, True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _action_summary(self, tool: ToolName, args: dict) -> str:
        if tool == ToolName.read_file:
            return f"read_file({args.get('filename', '?')})"
        elif tool == ToolName.search_code:
            q = args.get("query", "?")
            return f"search_code({q[:30]})"
        elif tool == ToolName.run_tests:
            return "run_tests()"
        elif tool == ToolName.inspect_function:
            return f"inspect_function({args.get('function_name', '?')})"
        elif tool == ToolName.submit_root_cause:
            return f"submit_root_cause(file={args.get('filename', '?')}, line={args.get('line', '?')})"
        return str(tool)

    def _error_obs(self, message: str, reward: float) -> BugsBuddyObservation:
        return BugsBuddyObservation(
            done=False,
            reward=reward,
            tool_output=message,
            tool_success=False,
            available_files=[],
            steps_remaining=0,
            action_history=[],
        )

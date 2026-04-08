"""Bugs Buddy Environment Client."""

from typing import Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import BugsBuddyAction, BugsBuddyObservation, BugReport, BugsBuddyState, ToolName


class BugsBuddyEnv(EnvClient[BugsBuddyAction, BugsBuddyObservation, BugsBuddyState]):
    """
    Async WebSocket client for the Bugs Buddy Environment.

    Example:
        async with BugsBuddyEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(task_id="task_easy")
            obs = result.observation
            while not result.done:
                action = BugsBuddyAction(tool=ToolName.run_tests, args={})
                result = await env.step(action)
    """

    def _step_payload(self, action: BugsBuddyAction) -> Dict:
        return {
            "tool": action.tool.value,
            "args": action.args,
        }

    def _parse_result(self, payload: Dict) -> StepResult[BugsBuddyObservation]:
        obs_data = payload.get("observation", {})

        # Parse nested BugReport if present
        bug_report = None
        br_data = obs_data.get("bug_report")
        if br_data:
            bug_report = BugReport(
                title=br_data.get("title", ""),
                description=br_data.get("description", ""),
                stack_trace=br_data.get("stack_trace"),
                task_id=br_data.get("task_id", ""),
                difficulty=br_data.get("difficulty", ""),
            )

        observation = BugsBuddyObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            tool_output=obs_data.get("tool_output", ""),
            tool_success=obs_data.get("tool_success", True),
            bug_report=bug_report,
            available_files=obs_data.get("available_files", []),
            steps_remaining=obs_data.get("steps_remaining", 20),
            action_history=obs_data.get("action_history", []),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> BugsBuddyState:
        return BugsBuddyState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", ""),
            files_read=payload.get("files_read", []),
            tests_run=payload.get("tests_run", False),
            current_hypothesis=payload.get("current_hypothesis"),
        )

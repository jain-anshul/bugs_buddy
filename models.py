# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Bugs Buddy RL Environment.

The agent investigates a small Python codebase using investigative tools
and submits a root cause hypothesis for grading.
"""

from enum import Enum
from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class ToolName(str, Enum):
    read_file = "read_file"
    search_code = "search_code"
    run_tests = "run_tests"
    inspect_function = "inspect_function"
    submit_root_cause = "submit_root_cause"


class BugsBuddyAction(Action):
    """Agent action: call one of the 5 investigative tools."""

    tool: ToolName = Field(..., description="Tool to invoke")
    args: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Tool arguments. "
            "read_file: {filename}, "
            "search_code: {query}, "
            "run_tests: {}, "
            "inspect_function: {function_name}, "
            "submit_root_cause: {filename, line, explanation}"
        ),
    )


class BugReport(BaseModel):
    """Bug report presented to the agent at the start of an episode."""

    title: str = Field(..., description="One-line bug title")
    description: str = Field(..., description="Full description with observed vs expected behaviour")
    stack_trace: Optional[str] = Field(default=None, description="Stack trace if available")
    task_id: str = Field(..., description="Task identifier")
    difficulty: str = Field(..., description="easy, medium, or hard")


class BugsBuddyObservation(Observation):
    """Observation returned after each step."""

    tool_output: str = Field(default="", description="Result of the action: file content, search results, test output, or grader feedback")
    tool_success: bool = Field(default=True, description="Whether the action was valid and executed cleanly")
    bug_report: Optional[BugReport] = Field(default=None, description="Bug report (constant across episode)")
    available_files: List[str] = Field(default_factory=list, description="All filenames in the bundled codebase")
    steps_remaining: int = Field(default=20, description="Steps remaining before forced termination")
    action_history: List[str] = Field(default_factory=list, description="Short summary of each prior action")
    grader_score: Optional[float] = Field(default=None, description="Raw grader score [0,1] set only on submit_root_cause; None for all other steps")


class BugsBuddyState(State):
    """Internal episode state (not part of observation)."""

    task_id: str = Field(default="", description="Active task identifier")
    difficulty: str = Field(default="", description="Task difficulty level")
    files_read: List[str] = Field(default_factory=list, description="Files accessed this episode")
    tests_run: bool = Field(default=False, description="Whether run_tests has been called")
    current_hypothesis: Optional[str] = Field(default=None, description="Most recently submitted hypothesis")

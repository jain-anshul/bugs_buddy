"""Bundled task definitions for Bugs Buddy."""

from .task_easy import TASK_EASY
from .task_medium import TASK_MEDIUM
from .task_hard import TASK_HARD

ALL_TASKS = {
    "task_easy": TASK_EASY,
    "task_medium": TASK_MEDIUM,
    "task_hard": TASK_HARD,
}

__all__ = ["TASK_EASY", "TASK_MEDIUM", "TASK_HARD", "ALL_TASKS"]

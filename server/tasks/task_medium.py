"""
Task Medium: Silent Wrong Result from Incorrect Operator Precedence
--------------------------------------------------------------------
Difficulty: Medium | 3 files | ~150 lines | Expected steps: 6-12

The bug is in compare_groups() in aggregator.py: the expression
`mean_a / mean_b * 100` is evaluated as `mean_a / (mean_b * 100)` due to
left-to-right evaluation — the result is 100x smaller than intended.
No test covers compare_groups() directly, so all tests pass.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GroundTruth:
    filename: str
    line: int
    buggy_function: str
    relevant_files: List[str]
    keywords: List[str]


@dataclass
class TaskDefinition:
    task_id: str
    difficulty: str
    bug_report_title: str
    bug_report_description: str
    bug_report_stack_trace: str | None
    files: Dict[str, str]
    test_output: str
    ground_truth: GroundTruth


# ---------------------------------------------------------------------------
# Bundled codebase
# ---------------------------------------------------------------------------

_STATS_PY = '''\
# analytics/stats.py
"""Statistical computation utilities."""

import math


def mean(values):
    """Compute the arithmetic mean of a list of numbers."""
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def variance(values):
    """Compute the population variance of a list of numbers."""
    if len(values) < 2:
        raise ValueError("Need at least 2 values for variance")
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def std_deviation(values):
    """Compute the standard deviation of a list of numbers."""
    return math.sqrt(variance(values))


def median(values):
    """Compute the median of a list of numbers."""
    if not values:
        raise ValueError("Cannot compute median of empty list")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]
'''

_AGGREGATOR_PY = '''\
# analytics/aggregator.py
"""Dataset aggregation and group comparison utilities."""

try:
    from .stats import mean, std_deviation
except ImportError:
    from stats import mean, std_deviation


def summarize_dataset(data):
    """
    Summarize a dataset dict {group_name: [values]}.
    Returns {group_name: {mean, std, count}}.
    """
    summary = {}
    for group, values in data.items():
        summary[group] = {
            "mean": mean(values),
            "std": std_deviation(values),
            "count": len(values),
        }
    return summary


def compare_groups(data_a, data_b):
    """
    Compare two datasets and return the ratio of their means as a percentage.

    Args:
        data_a: list of numeric values for group A
        data_b: list of numeric values for group B

    Returns:
        float: (mean_a / mean_b) * 100  — i.e. mean_a as % of mean_b
    """
    mean_a = mean(data_a)
    mean_b = mean(data_b)
    if mean_b == 0:
        raise ZeroDivisionError("Group B mean is zero; cannot compute ratio")
    return mean_a / mean_b * 100  # BUG: evaluated as mean_a / (mean_b * 100)


def top_n(values, n=3):
    """Return the top-n largest values from a list."""
    return sorted(values, reverse=True)[:n]


def bottom_n(values, n=3):
    """Return the bottom-n smallest values from a list."""
    return sorted(values)[:n]
'''

_FORMATTER_PY = '''\
# analytics/formatter.py
"""Report formatting utilities for analytics output."""


def round_values(d, decimals=2):
    """Recursively round all numeric values in a dict."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = round_values(v, decimals)
        elif isinstance(v, float):
            result[k] = round(v, decimals)
        else:
            result[k] = v
    return result


def format_report(summary, title="Dataset Summary"):
    """
    Format a dataset summary dict into a human-readable string.

    Args:
        summary: dict from summarize_dataset()
        title: report header string

    Returns:
        str: formatted report
    """
    lines = [f"=== {title} ==="]
    for group, stats in summary.items():
        lines.append(f"  {group}:")
        lines.append(f"    mean  = {stats['mean']:.4f}")
        lines.append(f"    std   = {stats['std']:.4f}")
        lines.append(f"    count = {stats['count']}")
    return "\n".join(lines)


def format_comparison(ratio, group_a="A", group_b="B"):
    """Format a compare_groups() ratio for display."""
    return f"{group_a} is {ratio:.1f}% of {group_b}"
'''

# Bug is on line 41 of aggregator.py:
#   return mean_a / mean_b * 100  # BUG

_TEST_OUTPUT = """\
test_mean ... ok
test_variance ... ok
test_std_deviation ... ok
test_median ... ok
test_summarize_dataset ... ok
test_format_report ... ok
test_round_values ... ok
test_top_n ... ok
test_bottom_n ... ok

----------------------------------------------------------------------
Ran 9 tests in 0.002s

OK

NOTE: No direct test for compare_groups() exists.
      The function is called by downstream dashboard code.
      All unit tests pass despite the bug.
"""

TASK_MEDIUM = TaskDefinition(
    task_id="task_medium",
    difficulty="medium",
    bug_report_title="compare_groups() returns ratios that are 100x too small",
    bug_report_description=(
        "compare_groups() in aggregator.py returns incorrect comparison ratios "
        "for all datasets. No exception is raised and no test currently covers "
        "this function directly. The bug was discovered when a downstream "
        "dashboard showed all group ratios as approximately 0.01x instead of "
        "the expected ~1.0x (i.e. 100%). For two groups with equal means, "
        "compare_groups() should return 100.0, but it returns 1.0. The error "
        "appears to be in the calculation logic of compare_groups(), not in "
        "mean() or the formatting layer."
    ),
    bug_report_stack_trace=None,
    files={
        "analytics/stats.py": _STATS_PY,
        "analytics/aggregator.py": _AGGREGATOR_PY,
        "analytics/formatter.py": _FORMATTER_PY,
    },
    test_output=_TEST_OUTPUT,
    ground_truth=GroundTruth(
        filename="analytics/aggregator.py",
        line=41,
        buggy_function="compare_groups",
        relevant_files=["analytics/aggregator.py", "analytics/stats.py"],
        keywords=["precedence", "parentheses", "order", "division", "100x", "ratio", "operator"],
    ),
)

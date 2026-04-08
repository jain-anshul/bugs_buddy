"""
Task Easy: Wrong Return Value in a Discount Calculator
------------------------------------------------------
Difficulty: Easy | 1 file | ~40 lines | Expected steps: 2-4

The bug is in get_final_price(): it adds the discount amount instead of
subtracting it, so a 100% discount returns the original price unchanged.
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
    files: Dict[str, str]          # filename -> source code
    test_output: str               # pre-computed, deterministic
    ground_truth: GroundTruth


# ---------------------------------------------------------------------------
# Bundled codebase
# ---------------------------------------------------------------------------

_PRICING_PY = '''\
# store/pricing.py
"""
Pricing utilities for the online store module.
Provides discount calculation and price formatting functions.
"""


TAX_RATE = 0.08
DEFAULT_CURRENCY = "USD"
MIN_DISCOUNT = 0
MAX_DISCOUNT = 100


def calculate_discount(price, discount_percent):
    """Calculate the discount amount from a price and percentage."""
    if discount_percent < MIN_DISCOUNT or discount_percent > MAX_DISCOUNT:
        raise ValueError(
            f"Discount must be between {MIN_DISCOUNT} and {MAX_DISCOUNT}"
        )
    return price * (discount_percent / 100)


def apply_discount(price, discount_amount):
    """Apply a pre-calculated discount amount to a price."""
    return price - discount_amount


def get_final_price(price, discount_percent):
    """Get the final price after applying a percentage discount."""
    discount_amount = calculate_discount(price, discount_percent)
    return price + discount_amount


def format_price(price, currency=DEFAULT_CURRENCY):
    """Format a price for display."""
    return f"{currency} {price:.2f}"


def bulk_discount(items, discount_percent):
    """Apply a percentage discount to a list of (name, price) tuples."""
    results = []
    for name, price in items:
        final = get_final_price(price, discount_percent)
        results.append((name, final))
    return results
'''

# Bug is at line 32 of store/pricing.py (1-indexed)

_TEST_OUTPUT = """\
test_full_discount ... FAIL
AssertionError: Expected 0.0, got 150.0

test_partial_discount ... FAIL
AssertionError: Expected 135.0, got 165.0

test_zero_discount ... ok

test_format_price ... ok

test_bulk_discount ... FAIL
AssertionError: Expected [('shirt', 45.0)], got [('shirt', 105.0)]

----------------------------------------------------------------------
Ran 5 tests in 0.001s

FAILED (failures=3)
"""

TASK_EASY = TaskDefinition(
    task_id="task_easy",
    difficulty="easy",
    bug_report_title="get_final_price() returns higher price after discount",
    bug_report_description=(
        "get_final_price() returns a price that is *higher* than the original "
        "when any non-zero discount is applied. For a 100% discount on a $150 "
        "item, the expected result is $0.00 but the function returns $150.00 "
        "(the original price unchanged at 0% discount) — actually for 100% "
        "discount it returns $300.00. For a 10% discount on $150, expected is "
        "$135.00 but got $165.00. No exception is raised. The bug appears to "
        "be in the price calculation logic, not the discount amount computation."
    ),
    bug_report_stack_trace=None,
    files={
        "store/pricing.py": _PRICING_PY,
    },
    test_output=_TEST_OUTPUT,
    ground_truth=GroundTruth(
        filename="store/pricing.py",
        line=32,
        buggy_function="get_final_price",
        relevant_files=["store/pricing.py"],
        keywords=["addition", "subtraction", "subtract", "plus", "minus", "discount", "operator", "wrong operator", "+ should be -"],
    ),
)

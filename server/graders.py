"""
Deterministic graders for the three Bugs Buddy tasks.

Each grader scores a submitted root cause hypothesis in [0.0, 1.0] based on:
  - Filename correctness
  - Line number proximity
  - Keyword presence in explanation
  - (Medium/Hard) function name mentions
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tasks.task_easy import GroundTruth


def _keyword_hits(explanation: str, keywords: list[str]) -> int:
    """Count how many keywords appear in the explanation (case-insensitive)."""
    lower = explanation.lower()
    return sum(1 for kw in keywords if kw.lower() in lower)


def grade_easy(filename: str, line: int, explanation: str, ground_truth: "GroundTruth") -> float:
    """
    Easy grader: discount calculator bug.

    Scoring:
      - 0.30  correct filename (exact)
      - 0.40  correct line ± 3
      - 0.30  ≥ 2 keyword hits from ground_truth.keywords
    """
    score = 0.0

    if filename == ground_truth.filename:
        score += 0.30

    if abs(line - ground_truth.line) <= 3:
        score += 0.40

    hits = _keyword_hits(explanation, ground_truth.keywords)
    if hits >= 2:
        score += 0.30
    elif hits == 1:
        score += 0.10  # partial credit

    return round(min(score, 1.0), 4)


def grade_medium(filename: str, line: int, explanation: str, ground_truth: "GroundTruth") -> float:
    """
    Medium grader: operator precedence bug.

    Scoring:
      - 0.25  correct filename (exact)
      - 0.25  correct line ± 5
      - 0.20  explanation mentions the buggy function name
      - 0.30  ≥ 2 keyword hits from ground_truth.keywords
    """
    score = 0.0

    if filename == ground_truth.filename:
        score += 0.25

    if abs(line - ground_truth.line) <= 5:
        score += 0.25

    if ground_truth.buggy_function.lower() in explanation.lower():
        score += 0.20

    hits = _keyword_hits(explanation, ground_truth.keywords)
    if hits >= 2:
        score += 0.30
    elif hits == 1:
        score += 0.10

    return round(min(score, 1.0), 4)


def grade_hard(filename: str, line: int, explanation: str, ground_truth: "GroundTruth") -> float:
    """
    Hard grader: pagination off-by-one bug.

    Scoring:
      - 0.20  correct filename (exact)
      - 0.20  correct line ± 8
      - 0.15  explanation mentions a secondary file (data_loader or fetch_batch)
      - 0.15  explanation mentions get_page or total_pages
      - 0.30  ≥ 3 keyword hits from ground_truth.keywords
    """
    score = 0.0

    if filename == ground_truth.filename:
        score += 0.20

    if abs(line - ground_truth.line) <= 8:
        score += 0.20

    lower = explanation.lower()

    # Secondary file mention
    if "data_loader" in lower or "fetch_batch" in lower:
        score += 0.15

    # Correct function mention — only rewarded when the correct file is identified,
    # since get_page/total_pages live in paginator.py
    if filename == ground_truth.filename and ("get_page" in lower or "total_pages" in lower):
        score += 0.15

    hits = _keyword_hits(explanation, ground_truth.keywords)
    if hits >= 3:
        score += 0.30
    elif hits == 2:
        score += 0.20
    elif hits == 1:
        score += 0.10

    return round(min(score, 1.0), 4)


# Dispatch map used by environment.py
GRADERS = {
    "task_easy": grade_easy,
    "task_medium": grade_medium,
    "task_hard": grade_hard,
}

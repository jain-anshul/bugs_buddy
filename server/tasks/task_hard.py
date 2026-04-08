"""
Task Hard: Off-by-One in Pagination Utility Causing Data Loss on Last Page
---------------------------------------------------------------------------
Difficulty: Hard | 4 files | ~220 lines | Expected steps: 10-18

The bug spans paginator.py and data_loader.py. get_page() passes the batch
size (always == page_size) to the boundary guard instead of the total record
count. This causes the last page to return [] when fetch_batch() is used,
because `page > total_pages(batch, page_size)` evaluates to True for the
last page even when records exist there.
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

_PAGINATOR_PY = '''\
# api/paginator.py
"""
Pagination utilities for the API layer.

Provides get_page(), total_pages(), and paginate() for slicing record lists
into fixed-size pages. Designed to work with data_loader.fetch_batch().
"""

import math


def total_pages(records, page_size):
    """Return the total number of pages for a given record list and page size."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return math.ceil(len(records) / page_size)


def get_page(records, page, page_size, total_record_count=None):
    """
    Return the records for a specific 1-indexed page.

    Args:
        records:            The (possibly partial) list of records to slice.
        page:               1-indexed page number to retrieve.
        page_size:          Maximum number of records per page.
        total_record_count: Total number of records in the full dataset.
                            Used for boundary checking. If None, defaults
                            to len(records).

    Returns:
        list: Records for the requested page, or [] if page is out of range.
    """
    count = total_record_count if total_record_count is not None else len(records)
    if page < 1 or page > total_pages(records, page_size):  # BUG: should use count not records
        return []
    start = (page - 1) * page_size
    end = start + page_size
    return records[start:end]


def paginate(records, page_size):
    """
    Yield all pages of records as a list of lists.

    Args:
        records:   Full list of records.
        page_size: Maximum number of records per page.

    Yields:
        list: One page of records at a time.
    """
    n = total_pages(records, page_size)
    for page in range(1, n + 1):
        yield get_page(records, page, page_size)
'''

_DATA_LOADER_PY = '''\
# api/data_loader.py
"""
Data loading utilities for the API layer.

load_records() fetches all records. fetch_batch() retrieves a single
page-sized batch from the backing store.
"""


_RECORD_STORE = [f"record_{i}" for i in range(1, 101)]  # 100 records


def load_records(filters=None):
    """
    Load all records, optionally applying a filter function.

    Args:
        filters: Optional callable(record) -> bool to filter records.

    Returns:
        list: All matching records.
    """
    records = list(_RECORD_STORE)
    if filters:
        records = [r for r in records if filters(r)]
    return records


def fetch_batch(page, page_size, total_count, filters=None):
    """
    Fetch a single page-sized batch of records.

    Args:
        page:        1-indexed page number to retrieve.
        page_size:   Number of records per page.
        total_count: Total number of records in the full dataset.
        filters:     Optional callable(record) -> bool.

    Returns:
        list: Records for the requested page.
    """
    all_records = load_records(filters)
    start = (page - 1) * page_size
    end = start + page_size
    batch = all_records[start:end]
    # Pass batch (not all_records) to get_page for the slice — caller passes
    # total_count separately to handle the boundary check correctly.
    try:
        from .paginator import get_page
    except ImportError:
        from paginator import get_page
    # BUG TRIGGER: get_page is called with batch as records and no total_record_count,
    # so the boundary guard uses len(batch) instead of total_count.
    return get_page(batch, 1, page_size)
'''

_QUERY_BUILDER_PY = '''\
# api/query_builder.py
"""Query construction utilities for filtering and sorting record lists."""


def build_query(filters=None, sort_key=None, sort_reverse=False):
    """
    Build a query config dict from optional filter and sort parameters.

    Args:
        filters:      Dict of {field: value} constraints (simple equality).
        sort_key:     Field name to sort by (string attribute on record).
        sort_reverse: Whether to reverse the sort order.

    Returns:
        dict: Query configuration.
    """
    return {
        "filters": filters or {},
        "sort_key": sort_key,
        "sort_reverse": sort_reverse,
    }


def apply_filters(records, filters):
    """
    Apply a dict of {field: value} equality filters to a list of record strings.
    For string records, only supports prefix-based field matching.
    """
    if not filters:
        return records
    result = []
    for record in records:
        match = True
        for field, value in filters.items():
            if not record.startswith(str(value)):
                match = False
                break
        if match:
            result.append(record)
    return result


def apply_sort(records, sort_key=None, sort_reverse=False):
    """Sort a list of records by an optional key."""
    if sort_key is None:
        return records
    try:
        return sorted(records, key=lambda r: getattr(r, sort_key, r), reverse=sort_reverse)
    except Exception:
        return sorted(records, reverse=sort_reverse)
'''

_RESPONSE_PY = '''\
# api/response.py
"""Response formatting utilities for the API layer."""


def format_response(page_data, page, page_size, total_count):
    """
    Wrap paginated data in a standard API response envelope.

    Args:
        page_data:   List of records for the current page.
        page:        Current 1-indexed page number.
        page_size:   Records per page.
        total_count: Total number of records in the full dataset.

    Returns:
        dict: Standard API response with data and metadata.
    """
    return {
        "data": page_data,
        "meta": attach_metadata(page, page_size, total_count, len(page_data)),
    }


def attach_metadata(page, page_size, total_count, returned_count):
    """
    Build the metadata section of a paginated API response.

    Args:
        page:           Current 1-indexed page number.
        page_size:      Records per page.
        total_count:    Total records in the dataset.
        returned_count: Number of records actually returned on this page.

    Returns:
        dict: Metadata with pagination info.
    """
    import math
    total = math.ceil(total_count / page_size) if page_size > 0 else 0
    return {
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "total_pages": total,
        "returned_count": returned_count,
        "has_next": page < total,
        "has_prev": page > 1,
    }
'''

# Bug is at line 36 of paginator.py:
#   if page < 1 or page > total_pages(records, page_size):  # BUG

_TEST_OUTPUT = """\
test_first_page ... ok
test_middle_page ... ok
test_last_page_full ... ok
test_last_page_partial ... FAIL
AssertionError: Expected ['record_9', 'record_10'], got []

test_single_page ... ok
test_empty_dataset ... ok
test_total_pages ... ok
test_format_response ... ok
test_attach_metadata ... ok

----------------------------------------------------------------------
Ran 9 tests in 0.003s

FAILED (failures=1)

NOTE: test_last_page_partial fails when fetch_batch() is used as the record
      source. Direct calls to get_page() with a full record list pass correctly.
      The failure only manifests when records are retrieved via fetch_batch().
"""

TASK_HARD = TaskDefinition(
    task_id="task_hard",
    difficulty="hard",
    bug_report_title="Last page of paginated results always returns empty list",
    bug_report_description=(
        "The last page of paginated results is always missing when pagination "
        "is driven through fetch_batch(). When a dataset has N records and "
        "page size is P, requesting page ceil(N/P) via fetch_batch() returns "
        "an empty list instead of the remaining records. All earlier pages "
        "return correctly. The bug is observed across all API endpoints that "
        "use fetch_batch() for pagination. Direct calls to get_page() with a "
        "full record list work correctly. No exception is raised — the last "
        "page silently returns []."
    ),
    bug_report_stack_trace=None,
    files={
        "api/paginator.py": _PAGINATOR_PY,
        "api/data_loader.py": _DATA_LOADER_PY,
        "api/query_builder.py": _QUERY_BUILDER_PY,
        "api/response.py": _RESPONSE_PY,
    },
    test_output=_TEST_OUTPUT,
    ground_truth=GroundTruth(
        filename="api/paginator.py",
        line=36,
        buggy_function="get_page",
        relevant_files=["api/paginator.py", "api/data_loader.py"],
        keywords=["count", "total", "batch", "boundary", "guard", "ceil", "off-by-one", "page_size", "last page"],
    ),
)

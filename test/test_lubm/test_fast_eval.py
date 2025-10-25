from __future__ import annotations

import os
import time

import pytest

from krrood.experiments.helpers import (
    load_instances_for_lubm_with_predicates,
    evaluate_eql,
)
from krrood.experiments.lubm_eql_queries import get_eql_queries
from krrood.entity_query_language.fast_eval import evaluate_fast
from krrood.entity_query_language.eql_to_python import compile_to_python
from krrood.entity_query_language.failures import NoSolutionFound


# @pytest.fixture(scope="module", autouse=True)
# def _setup_lubm_env():
#     # Enable the fast path and ensure the LUBM model/classes and cache are loaded once
#     old = os.environ.get("KRROOD_FAST_EVAL")
#     os.environ["KRROOD_FAST_EVAL"] = "1"
#     try:
#         load_instances_for_lubm_with_predicates()
#         yield
#     finally:
#         if old is None:
#             os.environ.pop("KRROOD_FAST_EVAL", None)
#         else:
#             os.environ["KRROOD_FAST_EVAL"] = old


def _count_fast(q):
    it = evaluate_fast(q)
    if it is None:
        pytest.skip("fast path not applicable for this query")
    return sum(1 for _ in it)


def test_fast_q6_student_scan_matches_reference():
    # q6 is a simple scan over Student()
    q6 = get_eql_queries()[5]
    fast_count = _count_fast(q6)
    counts, _, _ = evaluate_eql([q6])
    assert fast_count == counts[0]


def test_fast_q7_contains_with_precompute_matches_reference():
    # q7 is a join using contains with a fixed AssociateProfessor on the left and the student's courses on the right
    q7 = get_eql_queries()[6]
    fast_count = _count_fast(q7)
    # The reference interpreter may raise NoSolutionFound for The(...) wrappers; compile and run instead
    compiled = compile_to_python(q7)
    compiled_count = len(list(compiled.function()))
    assert fast_count == compiled_count


def test_fast_path_agrees_with_reference_when_applicable():
    registry = load_instances_for_lubm_with_predicates()
    start_time = time.time()
    counts, results, times = evaluate_eql(get_eql_queries())
    end_time = time.time()
    for i, n in enumerate(counts, 1):
        print(f"{i}:{n} ({times[i - 1]} sec)")
        print({type(r) for r in results[i - 1]})
    print(f"Time elapsed: {end_time - start_time} seconds")
    for i, q in enumerate(get_eql_queries()):
        it = evaluate_fast(q)
        if it is None:
            print("fast path not applicable for this query")
            continue
        start = time.time()
        fast_count = sum(1 for _ in it)
        end = time.time()
        print(f"Fast path took {end - start} seconds")
        assert fast_count == counts[i]

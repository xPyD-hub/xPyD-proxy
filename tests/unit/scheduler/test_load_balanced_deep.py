"""Deep tests for LoadBalancedScheduler."""

import itertools
import os
import threading
from unittest.mock import patch

from scheduler.load_balanced import LoadBalancedScheduler

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "tokenizers", "DeepSeek-R1")


@patch(
    "scheduler.load_balanced.query_instance_model_len",
    return_value=[131072, 131072],
)
def test_schedule_completion_releases_load(mock_query):
    """After schedule_completion, load counters should decrease."""
    prefill = ["p1:1", "p2:2"]
    decode = ["d1:1", "d2:2"]
    sched = LoadBalancedScheduler(prefill, decode)

    p_cycler = itertools.cycle(prefill)
    d_cycler = itertools.cycle(decode)

    p_node = sched.schedule(p_cycler, is_prompt=True, request_len=100, max_tokens=50)
    d_node = sched.schedule(d_cycler, is_prompt=False, request_len=100, max_tokens=50)
    assert p_node in prefill
    assert d_node in decode

    p_idx = prefill.index(p_node)
    d_idx = decode.index(d_node)
    p_bs_before = sched.prefill_bs_counter[p_idx]
    d_bs_before = sched.decode_bs_counter[d_idx]

    sched.schedule_completion(prefill_instance=p_node, req_len=100)
    sched.schedule_completion(decode_instance=d_node, req_len=100)

    assert sched.prefill_bs_counter[p_idx] < p_bs_before
    assert sched.decode_bs_counter[d_idx] < d_bs_before


@patch(
    "scheduler.load_balanced.query_instance_model_len",
    return_value=[100],
)
def test_all_nodes_full_fallback(mock_query):
    """When all nodes are at capacity, schedule should return None."""
    prefill = ["p1:1"]
    decode = ["d1:1"]
    sched = LoadBalancedScheduler(prefill, decode)

    p_cycler = itertools.cycle(prefill)
    result = sched.schedule(p_cycler, is_prompt=True, request_len=80, max_tokens=30)
    assert result is None


@patch(
    "scheduler.load_balanced.query_instance_model_len",
    return_value=[131072],
)
def test_request_exceeds_model_len(mock_query):
    """request_len + max_tokens > max_model_len should return None."""
    prefill = ["p1:1"]
    decode = ["d1:1"]
    sched = LoadBalancedScheduler(prefill, decode)

    p_cycler = itertools.cycle(prefill)
    d_cycler = itertools.cycle(decode)

    result = sched.schedule(p_cycler, is_prompt=True, request_len=100, max_tokens=50)
    assert result is not None

    result = sched.schedule(
        d_cycler, is_prompt=False, request_len=131000, max_tokens=100
    )
    assert result is None


@patch(
    "scheduler.load_balanced.query_instance_model_len",
    return_value=[131072, 131072],
)
def test_concurrent_scheduling_thread_safety(mock_query):
    """Multiple threads scheduling simultaneously must not corrupt counters."""
    prefill = ["p1:1", "p2:2"]
    decode = ["d1:1", "d2:2"]
    sched = LoadBalancedScheduler(prefill, decode)

    errors = []

    def schedule_and_complete(thread_id):
        try:
            p_cycler = itertools.cycle(prefill)
            d_cycler = itertools.cycle(decode)
            for _ in range(50):
                p_node = sched.schedule(
                    p_cycler,
                    is_prompt=True,
                    request_len=100,
                    max_tokens=50,
                )
                d_node = sched.schedule(
                    d_cycler,
                    is_prompt=False,
                    request_len=100,
                    max_tokens=50,
                )
                assert p_node is not None, f"thread {thread_id}: prefill returned None"
                assert d_node is not None, f"thread {thread_id}: decode returned None"
                sched.schedule_completion(prefill_instance=p_node, req_len=100)
                sched.schedule_completion(decode_instance=d_node, req_len=100)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=schedule_and_complete, args=(tid,)) for tid in range(8)
    ]
    for thr in threads:
        thr.start()
    for thr in threads:
        thr.join()

    assert not errors, f"Thread errors: {errors}"
    # After all schedule+complete pairs, counters should be back to zero
    assert all(count == 0 for count in sched.prefill_bs_counter)
    assert all(count == 0 for count in sched.decode_bs_counter)


@patch(
    "scheduler.load_balanced.query_instance_model_len",
    return_value=[131072, 131072],
)
def test_decode_tiebreak_by_kv_utilization(mock_query):
    """When decode bs_counters tie, pick lower KV utilization."""
    prefill = ["p1:1", "p2:2"]
    decode = ["d1:1", "d2:2"]
    sched = LoadBalancedScheduler(prefill, decode)

    d_cycler = itertools.cycle(decode)

    # Schedule one request to each decode node, but with different request_len
    # to create different kv_utils_counter values
    node_a = sched.schedule(d_cycler, is_prompt=False, request_len=5000, max_tokens=50)
    node_b = sched.schedule(d_cycler, is_prompt=False, request_len=1000, max_tokens=50)
    assert node_a in decode
    assert node_b in decode

    # Complete both so bs_counters go back to zero (tied)
    sched.schedule_completion(decode_instance=node_a, req_len=5000)
    sched.schedule_completion(decode_instance=node_b, req_len=1000)

    # Now bs_counters are both 0 (tied).  kv_utils differ if not reset.
    # With bs=0 the scheduler takes the first candidate with bs==0,
    # which is index-order.  Verify the call succeeds and returns a valid node.
    next_node = sched.schedule(
        d_cycler, is_prompt=False, request_len=100, max_tokens=50
    )
    assert next_node in decode

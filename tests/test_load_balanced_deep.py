"""Deep tests for LoadBalancedScheduler."""

import itertools
import os
from unittest.mock import patch

from MicroPDProxyServer import LoadBalancedScheduler

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "tokenizers", "DeepSeek-R1")


@patch("MicroPDProxyServer.query_instance_model_len", return_value=[131072, 131072])
def test_schedule_completion_releases_load(mock_query):
    """After schedule_completion, load counters should decrease."""
    prefill = ["p1:1", "p2:2"]
    decode = ["d1:1", "d2:2"]
    sched = LoadBalancedScheduler(prefill, decode)

    p_cycler = itertools.cycle(prefill)
    d_cycler = itertools.cycle(decode)

    # Schedule a request
    p = sched.schedule(p_cycler, is_prompt=True, request_len=100, max_tokens=50)
    d = sched.schedule(d_cycler, is_prompt=False, request_len=100, max_tokens=50)
    assert p in prefill
    assert d in decode

    # Record counters before completion
    p_idx = prefill.index(p)
    d_idx = decode.index(d)
    p_bs_before = sched.prefill_bs_counter[p_idx]
    d_bs_before = sched.decode_bs_counter[d_idx]

    # Complete
    sched.schedule_completion(prefill_instance=p, req_len=100)
    sched.schedule_completion(decode_instance=d, req_len=100)

    assert sched.prefill_bs_counter[p_idx] < p_bs_before
    assert sched.decode_bs_counter[d_idx] < d_bs_before


@patch("MicroPDProxyServer.query_instance_model_len", return_value=[100])
def test_all_nodes_full_fallback(mock_query):
    """When all nodes are at capacity, schedule should return None."""
    prefill = ["p1:1"]
    decode = ["d1:1"]
    sched = LoadBalancedScheduler(prefill, decode)

    p_cycler = itertools.cycle(prefill)

    # request_len + max_tokens > model_len → should return None
    result = sched.schedule(p_cycler, is_prompt=True, request_len=80, max_tokens=30)
    assert result is None


@patch("MicroPDProxyServer.query_instance_model_len", return_value=[131072])
def test_request_exceeds_model_len(mock_query):
    """request_len + max_tokens > max_model_len should return None."""
    prefill = ["p1:1"]
    decode = ["d1:1"]
    sched = LoadBalancedScheduler(prefill, decode)

    p_cycler = itertools.cycle(prefill)
    d_cycler = itertools.cycle(decode)

    # Prefill: fits
    r = sched.schedule(p_cycler, is_prompt=True, request_len=100, max_tokens=50)
    assert r is not None

    # Decode: exceeds
    r = sched.schedule(d_cycler, is_prompt=False, request_len=131000, max_tokens=100)
    assert r is None

"""Mutation checks: a cycling policy cannot pass busy-node semantics."""

import runpy
from pathlib import Path

import pytest

HARNESS = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "examples/lib/scheduler_semantics.py")
)


def test_busy_assertion_rejects_round_robin():
    with pytest.raises(AssertionError):
        HARNESS["assert_avoids_busy"](
            [{"decode": "idle"}, {"decode": "busy"}, {"decode": "idle"}],
            {"decode": "busy"},
        )


def test_busy_assertion_checks_both_roles():
    with pytest.raises(AssertionError):
        HARNESS["assert_avoids_busy"](
            [{"prefill": "busy-p", "decode": "idle-d"}] * 4,
            {"prefill": "busy-p", "decode": "busy-d"},
        )


def test_busy_assertion_accepts_repeated_idle_choice():
    HARNESS["assert_avoids_busy"](
        [{"prefill": "idle-p", "decode": "idle-d"}] * 4,
        {"prefill": "busy-p", "decode": "busy-d"},
    )


@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("strategy", HARNESS["STRATEGIES"])
def test_real_proxy_contract(topology, strategy):
    HARNESS["run"](topology, strategy)


@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("strategy", ["loadbalanced", "power_of_two"])
def test_real_proxy_rejects_round_robin_mutation(topology, strategy):
    with pytest.raises(AssertionError) as failure:
        HARNESS["run"](topology, strategy, actual_strategy="roundrobin")
    assert isinstance(failure.value.args[0], tuple)


@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("strategy", ["consistent_hash", "cache_aware"])
@pytest.mark.parametrize("replacement", ["loadbalanced", "roundrobin"])
def test_affinity_rejects_wrong_policy(topology, strategy, replacement):
    with pytest.raises(AssertionError) as failure:
        HARNESS["run"](topology, strategy, actual_strategy=replacement)
    assert failure.value.args[0][0] == "affinity-routing"

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for NodeDiscovery dual instance support."""

from __future__ import annotations

from xpyd.discovery import NodeDiscovery


class TestDiscoveryDualReady:
    def test_dual_only_is_ready(self):
        """All-dual deployment: is_ready=True when dual nodes are healthy."""
        d = NodeDiscovery(
            prefill_instances=[],
            decode_instances=[],
            dual_instances=["10.0.0.1:8000"],
        )
        d.healthy_dual.add("10.0.0.1:8000")
        assert d.is_ready is True

    def test_no_nodes_not_ready(self):
        """No healthy nodes: is_ready=False."""
        d = NodeDiscovery(
            prefill_instances=[],
            decode_instances=[],
            dual_instances=["10.0.0.1:8000"],
        )
        assert d.is_ready is False

    def test_pd_only_is_ready(self):
        """P/D only: is_ready when 1P+1D healthy."""
        d = NodeDiscovery(
            prefill_instances=["10.0.0.1:8000"],
            decode_instances=["10.0.0.2:8000"],
        )
        d.healthy_prefill.add("10.0.0.1:8000")
        d.healthy_decode.add("10.0.0.2:8000")
        assert d.is_ready is True

    def test_pd_missing_decode_not_ready(self):
        """P/D missing decode: not ready."""
        d = NodeDiscovery(
            prefill_instances=["10.0.0.1:8000"],
            decode_instances=["10.0.0.2:8000"],
        )
        d.healthy_prefill.add("10.0.0.1:8000")
        assert d.is_ready is False

    def test_mixed_dual_and_pd(self):
        """Mixed: dual healthy but P/D not complete → still ready (dual suffices)."""
        d = NodeDiscovery(
            prefill_instances=["10.0.0.1:8000"],
            decode_instances=[],
            dual_instances=["10.0.0.3:8000"],
        )
        d.healthy_dual.add("10.0.0.3:8000")
        assert d.is_ready is True

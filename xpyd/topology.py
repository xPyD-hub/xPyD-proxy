# SPDX-License-Identifier: Apache-2.0
"""Topology expansion: convert node + TP/DP parameters into instance endpoints.

Mirrors the logic in ``xpyd_start_proxy.sh:build_instance_endpoints``.
"""

from __future__ import annotations

from typing import Dict, List


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def validate_topology(
    role: str,
    nodes: List[str],
    tp_size: int,
    dp_size: int,
    world_size_per_node: int,
) -> None:
    """Validate topology parameters. Raises ``ValueError`` on failure."""
    for name, val in [
        ("tp_size", tp_size),
        ("dp_size", dp_size),
        ("world_size_per_node", world_size_per_node),
    ]:
        if not isinstance(val, int):
            raise ValueError(
                f"{role} {name} must be an integer, got {type(val).__name__}"
            )
    if tp_size < 1:
        raise ValueError(f"{role} tp_size must be a positive integer")
    if dp_size < 1:
        raise ValueError(f"{role} dp_size must be a positive integer")
    if world_size_per_node < 1:
        raise ValueError(f"{role} world_size_per_node must be a positive integer")
    if not _is_power_of_two(tp_size):
        raise ValueError(f"{role} tp_size must be a power of two, got {tp_size}")
    if not _is_power_of_two(dp_size):
        raise ValueError(f"{role} dp_size must be a power of two, got {dp_size}")
    if tp_size * dp_size != len(nodes) * world_size_per_node:
        raise ValueError(
            f"{role} topology invalid: tp_size({tp_size}) * dp_size({dp_size}) "
            f"!= nodes({len(nodes)}) * world_size_per_node({world_size_per_node})"
        )


def expand_topology(
    role: str,
    nodes: List[str],
    tp_size: int,
    dp_size: int,
    world_size_per_node: int,
) -> List[str]:
    """Expand topology into a list of ``host:port`` instance endpoints.

    Each node entry is ``"ip:base_port"``.  Returns one endpoint per DP
    instance, following the same mapping as ``xpyd_start_proxy.sh``.
    """
    validate_topology(role, nodes, tp_size, dp_size, world_size_per_node)

    # Parse node entries into (ip, base_port) pairs
    parsed: List[tuple] = []
    for entry in nodes:
        parts = entry.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"{role} invalid node format: {entry}")
        ip, port_str = parts
        try:
            base_port = int(port_str)
        except ValueError as exc:
            raise ValueError(
                f"{role} invalid port in node {entry}: {exc}"
            ) from exc
        parsed.append((ip, base_port))

    node_instance_counts: Dict[int, int] = {}
    endpoints: List[str] = []

    for instance_index in range(dp_size):
        start_rank = instance_index * tp_size
        main_node = start_rank // world_size_per_node

        if main_node >= len(nodes):
            raise ValueError(
                f"{role} topology expansion failed: computed node index "
                f"{main_node} out of range (only {len(nodes)} nodes)"
            )

        local_index = node_instance_counts.get(main_node, 0)
        ip, base_port = parsed[main_node]
        port = base_port + local_index
        node_instance_counts[main_node] = local_index + 1
        endpoints.append(f"{ip}:{port}")

    return endpoints

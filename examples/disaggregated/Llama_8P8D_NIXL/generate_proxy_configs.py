#!/usr/bin/env python3
"""Generate xPyD configurations for every scheduler and first-token source."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
MODEL = "/workspace/Meta-Llama-3-8B-Instruct/"
SCHEDULERS = (
    "roundrobin",
    "loadbalanced",
    "consistent_hash",
    "power_of_two",
    "cache_aware",
)

prefill = [f"127.0.0.1:{8100 + index}" for index in range(8)]
decode = [f"127.0.0.1:{8200 + index}" for index in range(8)]

for source in ("prefill", "decode"):
    output_dir = ROOT / "proxy_configs" / source
    output_dir.mkdir(parents=True, exist_ok=True)
    for scheduler in SCHEDULERS:
        config = {
            "model": MODEL,
            "prefill": prefill,
            "decode": decode,
            "port": 8868,
            "log_level": "info",
            "scheduling": scheduler,
            "disaggregated_mode": "nixl",
            "first_token_source": source,
            "health_check": {
                "enabled": True,
                "interval_seconds": 2,
                "timeout_seconds": 2,
            },
            "startup": {
                "wait_timeout_seconds": 1200,
                "probe_interval_seconds": 2,
            },
        }
        output = output_dir / f"{scheduler}.yaml"
        output.write_text(
            "# Loopback placeholders only.\n"
            "# prefill entries are prefill backend hosts; decode entries are decode backend hosts.\n"
            "# No proxy host is required for NIXL transport.\n"
            + yaml.safe_dump(config, sort_keys=False)
        )
        print(output)

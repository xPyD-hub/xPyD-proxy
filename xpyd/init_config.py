# SPDX-License-Identifier: Apache-2.0
"""Generate a well-commented YAML configuration template for xpyd proxy."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = """\
# xpyd proxy configuration
# Docs: https://github.com/xPyD-hub/xPyD-proxy

# Required: model name served by this proxy
model: "my-model"

# Required: at least one decode instance
# Can be a flat list or topology dict
decode:
  - "10.0.0.1:8000"

# Optional: prefill instances (for disaggregated prefill/decode)
# prefill:
#   - "10.0.0.2:8000"

# Server port (default: 8000)
port: 8000

# Log level: debug | info | warning | error (default: warning)
log_level: "warning"

# Scheduling policy: loadbalanced | roundrobin | consistent_hash | power_of_two | cache_aware
scheduling: "loadbalanced"

# Generate first token on prefill node (default: false)
generator_on_p_node: false

# Startup probe settings
startup:
  wait_timeout_seconds: 600
  probe_interval_seconds: 10

# Health check configuration
health_check:
  enabled: false
  interval_seconds: 10.0
  timeout_seconds: 3.0

# Circuit breaker configuration
circuit_breaker:
  enabled: false
  failure_threshold: 5
  success_threshold: 2
  timeout_duration_seconds: 30
  window_duration_seconds: 60

# Retry / resilience configuration
retry:
  enabled: false
  max_retries: 2
  initial_backoff_ms: 100
  max_backoff_ms: 10000
  backoff_multiplier: 2.0
  jitter_factor: 0.1
  retryable_status_codes: [408, 429, 500, 502, 503, 504]

# API keys (prefer env vars ADMIN_API_KEY / OPENAI_API_KEY)
# admin_api_key: ""
# openai_api_key: ""
"""


def generate_config_template(output_path: str) -> None:
    """Write a well-commented YAML template to *output_path*.

    Creates parent directories if they do not exist.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE)
    print(f"Config template written to {path}")

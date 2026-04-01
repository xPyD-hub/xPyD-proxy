> **Note:** Task designs and status tracking have moved to [GitHub Issues](https://github.com/xPyD-hub/xPyD-proxy/issues?q=label%3Atask). This document is retained only for the review policy and project overview.

# Task for OpenClaw

## Review Requirements (Strict Policy)

When reviewing PRs against this repository, bots and reviewers MUST adhere to the following strict guidelines:
1. **Line-by-line inspection:** Do not skim. Check every modified line.
2. **Robustness & Error Handling:** Look for unhandled exceptions, missing input validation (e.g., missing keys in JSON payloads), and edge case handling.
3. **Code Quality:** Reject hardcoded magic numbers, poor variable naming, and messy logic. Do not be lenient.
4. **Core Protection:** `core/MicroPDProxyServer.py` logic must remain intact and robust. If modifications are made, they must be purely formatting, safe pre-commit fixes, or necessary and robust bug fixes. Any regression in validation or core logic must result in a `REQUEST_CHANGES`.

---

## Project Overview

The most critical implementation file is the proxy server:
- <https://github.com/xPyD-hub/xPyD-proxy/blob/main/core/MicroPDProxyServer.py>

The system uses a shell script `xpyd_start_proxy.sh` to configure and launch a distributed proxy handling prefill and decode nodes.

---

## Tasks

All task specifications have been migrated to GitHub Issues:

- [#96 Task 1: Debug dummy nodes for proxy matrix configs](https://github.com/xPyD-hub/xPyD-proxy/issues/96)
- [#97 Task 2: Parameterize proxy shell script](https://github.com/xPyD-hub/xPyD-proxy/issues/97)
- [#98 Task 3: Benchmark validation with dummy nodes](https://github.com/xPyD-hub/xPyD-proxy/issues/98)
- [#99 Task 4: Refactor scheduler into core/scheduler/ module](https://github.com/xPyD-hub/xPyD-proxy/issues/99)
- [#100 Task 5: Add Prometheus-compatible /metrics endpoint](https://github.com/xPyD-hub/xPyD-proxy/issues/100)
- [#101 Task 6: Introduce core/config.py as single config source](https://github.com/xPyD-hub/xPyD-proxy/issues/101)
- [#102 Task 7: YAML-based configuration support](https://github.com/xPyD-hub/xPyD-proxy/issues/102)
- [#103 Task 8: CLI packaging (xpyd) and startup discovery](https://github.com/xPyD-hub/xPyD-proxy/issues/103)
- [#104 Task 9: Resilience — circuit breaker, retry, health monitor](https://github.com/xPyD-hub/xPyD-proxy/issues/104)
- [#105 Task 10: Advanced load balancing strategies](https://github.com/xPyD-hub/xPyD-proxy/issues/105)

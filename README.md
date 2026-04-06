# xPyD-proxy

**Lightweight Prefill-Decode disaggregated proxy for LLM serving.**

xPyD-proxy routes inference requests between prefill and decode nodes, enabling PD-disaggregated LLM serving with load balancing, health monitoring, and fault tolerance.

## Key Features

- **PD disaggregation** — separate prefill and decode nodes for optimal resource utilization
- **Multiple scheduling policies** — round-robin, consistent hash, cache-aware, power-of-two
- **Resilience** — circuit breaker, health monitoring, automatic failover
- **Multi-model routing** — serve multiple models through a single proxy
- **OpenAI-compatible API** — drop-in replacement for vLLM/OpenAI endpoints
- **YAML configuration** — declarative topology and settings

## Install

```bash
pip install xpyd-proxy
```

Or as part of the full xPyD toolkit:

```bash
pip install xpyd
```

## Quick Start

```bash
# Start with YAML config
xpyd proxy --config proxy.yaml

# Or with CLI args
xpyd proxy --model my-model \
  --prefill 127.0.0.1:8001 \
  --decode 127.0.0.1:8002
```

## Part of xPyD

| Component | Description |
|-----------|-------------|
| **xpyd-proxy** | PD-disaggregated proxy |
| [xpyd-sim](https://github.com/xPyD-hub/xPyD-sim) | OpenAI-compatible inference simulator |
| [xpyd-bench](https://github.com/xPyD-hub/xPyD-bench) | Benchmarking & planning tool |

📖 **[Full Guide →](docs/guide.md)** | 💡 **[Examples →](examples/)** | 🏗️ **[Contributing →](CONTRIBUTING.md)**

## License

Apache 2.0 — see [LICENSE](LICENSE)

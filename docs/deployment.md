# Deployment Guide

## System Requirements

- **Python** 3.10 or later
- **pip** (latest recommended)
- **Hardware:** GPU nodes running vLLM (or compatible) inference servers for
  prefill and decode roles
- **Network:** All nodes must be reachable from the proxy host

## Quick Install

```bash
git clone https://github.com/hlin99/MicroPDProxy.git
cd MicroPDProxy
pip install -r requirements.txt
```

## Running the Proxy

### Direct invocation

```bash
python3 core/MicroPDProxyServer.py \
  --model /path/to/tokenizer \
  --prefill 10.0.0.1:8100 10.0.0.2:8100 \
  --decode  10.0.0.3:8200 10.0.0.4:8200 \
  --port 8868
```

### Using `xpyd_start_proxy.sh`

For multi-node GPU deployments, use the parameterized launcher script:

```bash
export model_path=/shared/models/DeepSeek-R1

bash core/xpyd_start_proxy.sh \
  --prefill-nodes 2 --prefill-tp-size 8 --prefill-dp-size 2 --prefill-world-size-per-node 8 \
  --decode-nodes 2  --decode-tp-size 8  --decode-dp-size 2  --decode-world-size-per-node 8
```

See [`docs/xpyd_start_proxy_usage.md`](xpyd_start_proxy_usage.md) for full
parameter reference and topology rules.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ADMIN_API_KEY` | _(none)_ | API key for admin endpoints (`/instances/add`, `/instances/remove`). If unset, admin endpoints reject all requests. |
| `DUMMY_MODEL_ID` | _(none)_ | Override model ID returned by `/v1/models`. Useful for dummy/testing setups. |
| `XPYD_DECODE_IPS` | _(hardcoded list)_ | Space-separated decode node IPs for `xpyd_start_proxy.sh`. |
| `XPYD_PREFILL_IPS` | _(hardcoded list)_ | Space-separated prefill node IPs for `xpyd_start_proxy.sh`. |
| `XPYD_PROXY_PORT` | `8868` | Proxy listen port for `xpyd_start_proxy.sh`. |
| `XPYD_LOG` | _(none)_ | Directory for log files. If set, output is tee'd to a timestamped log. |
| `XPYD_DRY_RUN` | `0` | Set to `1` to print the command without executing. |

## Docker Deployment

```bash
# Build
docker build -t micropdproxy .

# Run (connect to external prefill/decode nodes)
docker run -p 8868:8868 micropdproxy \
  python3 core/MicroPDProxyServer.py \
  --model tokenizers/DeepSeek-R1 \
  --prefill 10.0.0.1:8100 \
  --decode  10.0.0.3:8200 \
  --port 8868
```

For a full local topology with dummy nodes, see the included
`docker-compose.yml`.

## Production Topology

A typical production deployment:

```
                    Load Balancer (:443)
                          │
                    ┌─────▼─────┐
                    │   Proxy   │
                    │  (:8868)  │
                    └──┬─────┬──┘
            ┌──────────┘     └──────────┐
     Prefill Pool                Decode Pool
  ┌──────────────────┐    ┌──────────────────┐
  │ Node A  (TP=8)   │    │ Node C  (TP=8)   │
  │ Node B  (TP=8)   │    │ Node D  (TP=8)   │
  └──────────────────┘    └──────────────────┘
```

## Monitoring and Logging

- **Health:** `GET /health` returns per-node health status
- **Status:** `GET /status` shows current instance pools
- **Logs:** Set `XPYD_LOG` to enable file logging with timestamps
- **Liveness probe:** `GET /ping` for container orchestrators

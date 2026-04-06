# One-Click Dummy + Proxy Setup

This guide shows how to run a complete local setup without real hardware:

- start dummy prefill servers
- start dummy decode servers
- launch the proxy through `xpyd_start_proxy.sh`
- send requests through the proxy

This is the fastest way to validate the project end-to-end on one machine.

---

## What this setup does

You run everything locally on `127.0.0.1`:

1. one or more dummy prefill nodes
2. one or more dummy decode nodes
3. the real proxy server
4. test requests through the proxy

The shell script still computes topology and expands instance endpoints.
The dummy nodes simply replace real hardware endpoints.

---

## Required environment

Set `model_path` first.

You can use a local tokenizer directory for testing, for example:

```bash
export model_path=$PWD/tests/assets/dummy_tokenizer
```

Optional but recommended for local runs:

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
```

---

## Example: one-command local topology

This example starts:

- 2 prefill instances
- 2 decode instances
- proxy on port `8868`

### Step 1: start dummy prefill nodes

Open terminal A:

```bash
uvicorn dummy_nodes.prefill_node:app --host 127.0.0.1 --port 8100 --log-level warning
```

Open terminal B:

```bash
uvicorn dummy_nodes.prefill_node:app --host 127.0.0.1 --port 8101 --log-level warning
```

### Step 2: start dummy decode nodes

Open terminal C:

```bash
uvicorn dummy_nodes.decode_node:app --host 127.0.0.1 --port 8200 --log-level warning
```

Open terminal D:

```bash
uvicorn dummy_nodes.decode_node:app --host 127.0.0.1 --port 8201 --log-level warning
```

### Step 3: launch proxy through the shell script

Open terminal E:

```bash
export model_path=$PWD/tests/assets/dummy_tokenizer
export XPYD_PREFILL_IPS="127.0.0.1 127.0.0.1"
export XPYD_DECODE_IPS="127.0.0.1 127.0.0.1"
export XPYD_PROXY_PORT=8868
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

bash core/xpyd_start_proxy.sh \
  -pn 2 -pt 8 -pd 2 -pw 8 \
  -dn 2 -dt 8 -dd 2 -dw 8 \
  --prefill-base-port 8100 \
  --decode-base-port 8200
```

This will launch the real `MicroPDProxyServer.py` using endpoints derived from the script.

---

## Send a test request

Once the proxy is up, run:

```bash
curl http://127.0.0.1:8868/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$PWD'/tests/assets/dummy_tokenizer",
    "messages": [{"role": "user", "content": "Hello from local dummy setup"}],
    "max_tokens": 4,
    "stream": false
  }'
```

You should get a normal chat-completion style JSON response.

---

## Streaming request example

```bash
curl http://127.0.0.1:8868/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$PWD'/tests/assets/dummy_tokenizer",
    "messages": [{"role": "user", "content": "stream please"}],
    "max_tokens": 4,
    "stream": true
  }'
```

You should see SSE output ending with:

```text
data: [DONE]
```

---

## Dry-run mode

If you only want to inspect the generated proxy command without launching it:

```bash
export model_path=$PWD/tests/assets/dummy_tokenizer
export XPYD_PREFILL_IPS="127.0.0.1 127.0.0.1"
export XPYD_DECODE_IPS="127.0.0.1 127.0.0.1"
export XPYD_PROXY_PORT=8868
export XPYD_DRY_RUN=1

bash core/xpyd_start_proxy.sh \
  -pn 2 -pt 8 -pd 2 -pw 8 \
  -dn 2 -dt 8 -dd 2 -dw 8 \
  --prefill-base-port 8100 \
  --decode-base-port 8200
```

This prints the final command and exits.

---

## Recommended files to read

- `docs/xpyd_start_proxy_usage.md` — detailed parameter and topology rules
- `tests/test_xpyd_start_proxy_integration.py` — real local integration example driven by the shell script

---

## Notes

- `xpyd_start_proxy.sh` does **not** auto-start dummy nodes for you.
- You still need to start the prefill/decode dummy servers separately.
- What this guide gives you is the complete end-to-end workflow in one place.

# Terminal-by-Terminal Quickstart

This guide is designed for copy-paste usage.

Follow the terminals in order:

- Terminal A: dummy prefill #1
- Terminal B: dummy prefill #2
- Terminal C: dummy decode #1
- Terminal D: dummy decode #2
- Terminal E: proxy via `xpyd_start_proxy.sh`
- Terminal F: test requests

Use this when you want the most explicit step-by-step local setup.

---

## Terminal A — start dummy prefill #1

```bash
cd /path/to/MicroPDProxy
uvicorn dummy_nodes.prefill_node:app --host 127.0.0.1 --port 8100 --log-level warning
```

Expected result:
- process keeps running
- no immediate error

---

## Terminal B — start dummy prefill #2

```bash
cd /path/to/MicroPDProxy
uvicorn dummy_nodes.prefill_node:app --host 127.0.0.1 --port 8101 --log-level warning
```

Expected result:
- process keeps running
- no immediate error

---

## Terminal C — start dummy decode #1

```bash
cd /path/to/MicroPDProxy
uvicorn dummy_nodes.decode_node:app --host 127.0.0.1 --port 8200 --log-level warning
```

Expected result:
- process keeps running
- no immediate error

---

## Terminal D — start dummy decode #2

```bash
cd /path/to/MicroPDProxy
uvicorn dummy_nodes.decode_node:app --host 127.0.0.1 --port 8201 --log-level warning
```

Expected result:
- process keeps running
- no immediate error

---

## Terminal E — launch proxy through the shell script

```bash
cd /path/to/MicroPDProxy

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

Expected result:
- terminal prints a `Running: python3 ./MicroPDProxyServer.py ...` line
- proxy keeps running in foreground
- no startup validation error

---

## Terminal F — check proxy status

```bash
curl http://127.0.0.1:8868/status
```

Expected result:
- JSON response
- `prefill_node_count` should be `2`
- `decode_node_count` should be `2`

---

## Terminal F — send a non-streaming request

```bash
cd /path/to/MicroPDProxy

curl http://127.0.0.1:8868/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$PWD'/tests/assets/dummy_tokenizer",
    "messages": [{"role": "user", "content": "Hello from terminal-by-terminal quickstart"}],
    "max_tokens": 4,
    "stream": false
  }'
```

Expected result:
- JSON response with `object: chat.completion`
- `choices[0].message.role` should be `assistant`

---

## Terminal F — send a streaming request

```bash
cd /path/to/MicroPDProxy

curl http://127.0.0.1:8868/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$PWD'/tests/assets/dummy_tokenizer",
    "messages": [{"role": "user", "content": "stream please"}],
    "max_tokens": 4,
    "stream": true
  }'
```

Expected result:
- SSE chunks in terminal
- output ends with:

```text
data: [DONE]
```

---

## Optional — inspect the generated command without launching proxy

Use this if you only want to see the shell-script-generated proxy command.

```bash
cd /path/to/MicroPDProxy

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

Expected result:
- prints the `Running: ...` command
- exits immediately

---

## Related docs

- `docs/xpyd_start_proxy_usage.md`
- `docs/one_click_dummy_proxy_setup.md`

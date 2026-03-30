# xpyd_start_proxy.sh Usage

This document describes how to use `core/xpyd_start_proxy.sh` after task-two parameterization.

## Purpose

`xpyd_start_proxy.sh` generates the prefill/decode topology for the proxy and starts `MicroPDProxyServer.py` with the expanded instance endpoints.

It now supports explicit command-line parameters instead of relying only on fixed topology values.

---

## Command format

```bash
bash core/xpyd_start_proxy.sh \
  --prefill-nodes <n> \
  --prefill-tp-size <n> \
  --prefill-dp-size <n> \
  --prefill-world-size-per-node <n> \
  --decode-nodes <n> \
  --decode-tp-size <n> \
  --decode-dp-size <n> \
  --decode-world-size-per-node <n> \
  [--prefill-base-port <n>] \
  [--decode-base-port <n>] \
  [--mode advanced|basic|benchmark|benchmark_decode]
```

### Short aliases

```bash
bash core/xpyd_start_proxy.sh \
  -pn <n> -pt <n> -pd <n> -pw <n> \
  -dn <n> -dt <n> -dd <n> -dw <n>
```

---

## Parameters

### Prefill parameters

- `--prefill-nodes` / `-pn`
- `--prefill-tp-size` / `-pt`
- `--prefill-dp-size` / `-pd`
- `--prefill-world-size-per-node` / `-pw`

### Decode parameters

- `--decode-nodes` / `-dn`
- `--decode-tp-size` / `-dt`
- `--decode-dp-size` / `-dd`
- `--decode-world-size-per-node` / `-dw`

### Optional parameters

- `--prefill-base-port`
  - default: `8100`
- `--decode-base-port`
  - default: `8200`
- `--mode` / `-m`
  - supported values:
    - `advanced`
    - `basic`
    - `benchmark`
    - `benchmark_decode`

---

## Topology rules

### Basic definitions

- One instance = one TP group
- N TP shards form one instance
- Instance count = `dp_size`
- `tp_size` defines how many TP shards are inside one instance
- `dp_size` defines how many total instances exist

### Validation rules

The script validates the following conditions:

1. All input values must be positive integers
2. `tp_size` must be a power of two
3. `dp_size` must be a power of two
4. The following must hold:

```text
tp_size * dp_size == nodes * world_size_per_node
```

5. Requested node count must not exceed the available IP list length

---

## Endpoint mapping rules

### Node / IP / Port

- Each node corresponds to one IP
- One node may host multiple instances
- Different instances on the same node use:
  - the same IP
  - different ports
  - ports incremented by `+1`

### Instance exposure rule

One instance exposes only one `IP:PORT` to the proxy.

Even if the instance internally spans multiple nodes, only the main node endpoint is exposed externally.

Main node definition:
- the node containing rank0 of the instance
- equivalently, the first node assigned to the TP group

### Case 1: `tp_size <= world_size_per_node`

One node can host one or more complete TP groups.

Each node hosts:

```text
world_size_per_node / tp_size
```

instances.

### Case 2: `tp_size > world_size_per_node`

One TP group spans multiple nodes.

Each instance spans:

```text
tp_size / world_size_per_node
```

nodes.

Different node shards belonging to the same instance use the same port internally, but only the main node `IP:PORT` is passed to the proxy.

---

## Environment variables

### Required

- `model_path`
  - must be set before running the script
  - passed through to `MicroPDProxyServer.py --model`

Example:

```bash
export model_path=/path/to/model/or/tokenizer
```

### Optional test-only overrides

These are mainly intended for local integration tests with dummy nodes:

- `XPYD_PREFILL_IPS`
- `XPYD_DECODE_IPS`
- `XPYD_PROXY_PORT`
- `XPYD_DRY_RUN=1`

`XPYD_DRY_RUN=1` prints the final command and exits without launching the proxy.

---

## Example commands

### Example 1: simple same-node topology

```bash
export model_path=/path/to/model

bash core/xpyd_start_proxy.sh \
  -pn 2 -pt 4 -pd 4 -pw 8 \
  -dn 2 -dt 2 -dd 8 -dw 8
```

This means:
- prefill: 2 nodes, 4 instances total, TP size 4, world size 8 per node
- decode: 2 nodes, 8 instances total, TP size 2, world size 8 per node

### Example 2: cross-node TP group

```bash
export model_path=/path/to/model

bash core/xpyd_start_proxy.sh \
  -pn 2 -pt 16 -pd 1 -pw 8 \
  -dn 4 -dt 8 -dd 4 -dw 8
```

This means:
- prefill TP group spans 2 nodes
- decode has 4 instances, each with TP size 8
- proxy still receives one endpoint per instance

### Example 3: custom base ports

```bash
export model_path=/path/to/model

bash core/xpyd_start_proxy.sh \
  -pn 1 -pt 8 -pd 1 -pw 8 \
  -dn 1 -dt 8 -dd 1 -dw 8 \
  --prefill-base-port 9100 \
  --decode-base-port 9200
```

---

## Dry-run debugging

To inspect the generated command without launching the proxy:

```bash
export model_path=/path/to/model
export XPYD_DRY_RUN=1

bash core/xpyd_start_proxy.sh \
  -pn 2 -pt 4 -pd 4 -pw 8 \
  -dn 2 -dt 2 -dd 8 -dw 8
```

The script will print:

```text
Running: python3 ./MicroPDProxyServer.py ...
```

and then exit.

---

## Testing

The following test layers exist for this script change:

### Static / validation tests

- `tests/test_xpyd_start_proxy_sh.py`

Covers:
- valid topology expansion
- custom base ports
- invalid TP/DP values
- invalid topology products
- node count exceeding IP list length
- non-integer / zero / negative arguments

### Real integration tests with dummy nodes

- `tests/test_xpyd_start_proxy_integration.py`

Covers:
- launching dummy prefill and decode nodes locally
- starting the proxy via `xpyd_start_proxy.sh`
- sending real chat-completion requests through the proxy
- verifying end-to-end behavior without real hardware

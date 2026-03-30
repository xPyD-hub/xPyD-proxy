# Task for OpenClaw

## 1. Core reference and current usage

The most critical implementation file is:

- <https://github.com/hlin99/MicroPDProxy/blob/main/core/MicroPDProxyServer.py>

### Current usage

#### a. Adjust proxy server in `xpyd_start_proxy.sh`

```bash
# Adjust Prefill/Decode IPs
PREFILL_IPS=("10.239.129.9" "10.239.129.67" "10.239.129.21" "10.239.128.165" "10.239.128.244" "10.239.128.153")
DECODE_IPS=("10.239.129.81" "10.239.129.165" "10.239.129.67" "10.239.129.21")
```

#### b. Current command form

```bash
bash xpyd_start_proxy.sh x y z
```

Notes:
- `x` = prefill node count
- `y` = decode node count
- `z` = TP size
- each node currently has 8 world size

---

## 2. Current task

### Task three (IN PROGRESS)

Continue with task three described below. This is the current active implementation task.

---

## 3. Task three details

### Goal

Support a real benchmark-style validation flow using `vllm bench serve` semantics in a local dummy-based setup.

Reference real-world benchmark command:

```bash
vllm bench serve --host 10.239.129.239 --port 8868 --model /workspace/hf_models/Qwen3-0.6B-Base/ --dataset-name random --random-input-len 3000 --random-output-len 200 --num-prompts 100 --burstiness 100 --request-rate 3.6
```

For task three:

- install `vllm` first via `pip install vllm`
- rewrite host / port / model according to the actual local test environment
- make the benchmark run successfully against:
  - 2 dummy prefill instances
  - 2 dummy decode instances
  - 1 proxy server

### Required topology

- prefill: TP8, DP2
- decode: TP1, DP16

### Validation target

Task three is considered complete only if the benchmark can run through the proxy successfully under the topology above.

This is not only a command-generation task. It must be validated end-to-end with real running processes.

### Expected work items

- prepare local dummy prefill / decode topology matching the task-three requirement
- start proxy with the correct parameterized shell script inputs
- install and invoke `vllm bench serve` with locally adjusted arguments
- debug any remaining incompatibility in the proxy / dummy setup if the benchmark cannot run through

### Testing / verification

At minimum, verify:

- the benchmark command can connect to the proxy
- requests are accepted and processed through the proxy
- the dummy prefill / decode stack is sufficient to support the benchmark path
- any missing dummy behavior required by the benchmark flow is fixed if necessary

---

--------------------------

## 4. ~~Task two~~ (DONE)

Task two has been implemented and validated.

## 5. Task two details


---

## 3. Task three details

### Goal

Support a real benchmark-style validation flow using `vllm bench serve` semantics in a local dummy-based setup.

Reference real-world benchmark command:

```bash
vllm bench serve --host 10.239.129.239 --port 8868 --model /workspace/hf_models/Qwen3-0.6B-Base/ --dataset-name random --random-input-len 3000 --random-output-len 200 --num-prompts 100 --burstiness 100 --request-rate 3.6
```

For task three:

- install `vllm` first via `pip install vllm`
- rewrite host / port / model according to the actual local test environment
- make the benchmark run successfully against:
  - 2 dummy prefill instances
  - 2 dummy decode instances
  - 1 proxy server

### Required topology

- prefill: TP8, DP2
- decode: TP1, DP16

### Validation target

Task three is considered complete only if the benchmark can run through the proxy successfully under the topology above.

This is not only a command-generation task. It must be validated end-to-end with real running processes.

### Expected work items

- prepare local dummy prefill / decode topology matching the task-three requirement
- start proxy with the correct parameterized shell script inputs
- install and invoke `vllm bench serve` with locally adjusted arguments
- debug any remaining incompatibility in the proxy / dummy setup if the benchmark cannot run through

### Testing / verification

At minimum, verify:

- the benchmark command can connect to the proxy
- requests are accepted and processed through the proxy
- the dummy prefill / decode stack is sufficient to support the benchmark path
- any missing dummy behavior required by the benchmark flow is fixed if necessary

--------------------------

## 4. ~~Task two~~ (DONE)

Task two has been implemented and validated.

## 5. Task two details

### Goal

Modify only `core/xpyd_start_proxy.sh` so that the current fixed configuration becomes command-line-parameter driven.

### Required parameters

#### Prefill parameters
- `--prefill-nodes` / `-pn`
- `--prefill-tp-size` / `-pt`
- `--prefill-dp-size` / `-pd`
- `--prefill-world-size-per-node` / `-pw`

#### Decode parameters
- `--decode-nodes` / `-dn`
- `--decode-tp-size` / `-dt`
- `--decode-dp-size` / `-dd`
- `--decode-world-size-per-node` / `-dw`

#### Optional parameters
- `--prefill-base-port`
- `--decode-base-port`

### Target command form

```bash
.sh \
  --prefill-nodes $a \
  --prefill-tp-size $b \
  --prefill-dp-size $c \
  --prefill-world-size-per-node $d \
  --decode-nodes $e \
  --decode-tp-size $f \
  --decode-dp-size $g \
  --decode-world-size-per-node $h
```

Since the command is long, short aliases such as `-pn`, `-pt`, etc. should also be supported.

---

### Validation rules

The script must validate the input arguments.

### Required checks

1. All parameters must be positive integers.
2. `tp_size` and `dp_size` must be powers of two.
3. The following constraint must hold:

```text
tp_size * dp_size == nodes * world_size_per_node
```

4. The requested node count cannot exceed the available IP list length:

```text
nodes <= IP list length
```

---

### Topology and mapping rules

### 5.1 Basic definitions

- One instance = one TP group.
- N TP shards form one instance.
- Instance count = `dp_size`.
- `tp_size` defines how many TP shards are inside one instance.
- `dp_size` defines how many total instances exist.

### 5.2 Node / IP / Port rules

- Each node corresponds to one IP.
- One node may host multiple instances.
- Different instances on the same node use:
  - the same IP,
  - different ports,
  - with ports incrementing by `+1`.

### 5.3 Instance exposure rule for proxy

- One instance exposes only one `IP:PORT` to the proxy.
- Even if one instance internally spans multiple nodes, only the main node IP and the instance port are exposed externally.
- Main node definition:
  - the node containing rank0 of the instance,
  - equivalently, the first node assigned to that TP group.

---

### Mapping rules under different TP / world-size relationships

### Case 1: `tp_size <= world_size_per_node`

This means one node can host one or more complete TP groups.

Rules:
- Each node can host:

```text
world_size_per_node / tp_size
```

instances.

- Multiple instances on the same node use the same IP and different incrementing ports.

### Case 2: `tp_size > world_size_per_node`

This means one TP group must span multiple nodes.

Rules:
- Each instance spans:

```text
tp_size / world_size_per_node
```

nodes.

- Different node shards belonging to the same instance use the same port.
- Only the main node `IP:PORT` is exposed to the proxy.

---

### Implementation constraints

- Only modify `core/xpyd_start_proxy.sh`.
- Do not modify other core business logic files.
- Small non-business-logic adjustments are allowed if necessary, such as:
  - path fixes,
  - argument parsing helpers,
  - usage/help output.

---

### Testing requirements

### 8.1 Same testing requirement as task one

After implementing task two, testing should be added similar to task one.

At minimum, verify that the generated topology / endpoint expansion works correctly under representative valid configurations.

### 8.2 Additional negative tests

Additional tests must be added for invalid TP / DP combinations and invalid topology inputs.

The script should fail clearly and handle invalid input gracefully.

### Invalid cases that must be covered

- `tp_size` is not a power of two
- `dp_size` is not a power of two
- `tp_size * dp_size != nodes * world_size_per_node`
- `nodes > IP list length`
- non-integer argument values
- zero or negative argument values

### Expected behavior for invalid cases

- Exit with non-zero status
- Print a clear and understandable error message
- Do not generate partial / ambiguous proxy arguments

## 6. ~~Task one~~ (DONE)

The content in `core/` has already been validated on real hardware. The code is considered basically correct.

## 7. Task one details

The first task was:

- without modifying core business logic,
- debug `dummy_nodes`,
- and make `dummy_nodes` work correctly under the following proxy server configurations:

### Required configurations

- `bash xpyd_start_proxy.sh 1 2 1`
- `bash xpyd_start_proxy.sh 2 2 1`
- `bash xpyd_start_proxy.sh 1 2 2`
- `bash xpyd_start_proxy.sh 1 2 4`
- `bash xpyd_start_proxy.sh 1 2 8`
- `bash xpyd_start_proxy.sh 2 2 2`
- `bash xpyd_start_proxy.sh 2 4 1`
- `bash xpyd_start_proxy.sh 2 4 2`

After debugging, submit PRs.

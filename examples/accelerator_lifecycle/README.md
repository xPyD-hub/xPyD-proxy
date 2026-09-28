# Accelerator lifecycle regression

Run the repository proxy against real vLLM backends on four **idle, dedicated**
CUDA or Intel XPU devices (two suffice for `--topology aggregated`):

```bash
bash examples/accelerator_lifecycle/run_all.sh \
  --device xpu --model /workspace/Qwen3.5-0.8B
# NVIDIA:
bash examples/accelerator_lifecycle/run_all.sh \
  --device cuda --model /models/small-causal-model --devices 0 1 2 3
```

Use an environment with the appropriate accelerator build of vLLM, proxy
dependencies, and `xpu-smi` or `nvidia-smi`. No Docker launcher or installed
`xpyd` command is needed. The XPU memory reader uses the
`memory.used_mib.<tile>.current` JSON schema; unsupported telemetry fails
explicitly. Model weights must fit alongside a 512 MiB KV cache per backend.

`run_all.sh` executes aggregated (2A), direct (2P2D), and mixed (1P1D + 2A)
scenarios serially. `--topology` selects one. `xpyd.yaml` is the mixed template;
the driver writes each exact runtime config to the log directory. Mixed
models use distinct `--served-model-name` values even though weights are shared.
Health checks are always enabled.

Each scenario checks proxy-first startup and inference 503, discovery through
`/status/instances`, exact per-role heartbeat counts, real inference, one-node
loss for each role, and recovery at the same address. A surviving peer must
continue serving. In mixed mode a missing P or D must make only the P/D model
return 503; the aggregated model must still return its own model name and text.
Restarts wait for the former process, port, and accelerator memory to be released.

**Direct mode does not transfer KV caches.** These tests exercise the proxy P/D
HTTP lifecycle with real accelerator inference, not NIXL/LMCache transport,
performance, or semantic model accuracy. Existing connector examples remain
the transport checks. Inference may be slow while models compile/warm up.

Output contains phase separators. `logs/<timestamp>/` holds proxy/backend
logs, runtime YAML, and `results.json`; `--log-dir` overrides it. Failure exits
nonzero, including cleanup failure. Only owned process groups are terminated.
Before reusing devices the test waits until memory is within 64 MiB of the
measured baseline. Other workloads on selected devices can invalidate this
check: do not use shared/busy devices.

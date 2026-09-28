# Local GPU integration suite

This directory provides a CI-like, one-command wrapper around the GPU
scenarios under `examples/`. It runs scenarios serially, captures their output,
enforces a per-scenario timeout, verifies that ports and GPU processes are
released, prints a summary, and returns a non-zero exit code if any scenario
fails.

## Prerequisites

- At least four NVIDIA GPUs
- `nvidia-smi`, `curl`, and GNU `timeout`
- A Python environment containing this repository, vLLM, and the `xpyd`
  command
- A local model compatible with the selected scenarios

Run the default smoke suite:

```bash
bash tests/gpu/run.sh
```

The default model path is `/workspace/Meta-Llama-3-8B-Instruct/`. Use
`--model` to load weights from another directory:

```bash
bash tests/gpu/run.sh --model /models/Meta-Llama-3-8B-Instruct
```

The default suite covers aggregated routing policies, direct 2P2D, and a mixed
aggregated/disaggregated deployment. Results are written beneath
`tests/gpu/logs/<UTC timestamp>/`; `summary.tsv` is suitable for scripts and
individual scenario logs contain the full output.

Use `--list` to inspect all cases, `--case NAME` to select one or more cases,
or `--all` to include the LMCache and NIXL 2P2D and 8P8D matrices:

```bash
bash tests/gpu/run.sh --list
bash tests/gpu/run.sh --case direct --case mixed
bash tests/gpu/run.sh --all --timeout 120
```

LMCache and NIXL cases require their corresponding Python packages. The matrix
cases run 16 model processes across four GPUs and therefore need substantially
more GPU memory than the default suite. LMCache scenarios use CPU transfer
buffers so the validation does not compete with model weights for GPU memory.
The LMCache 8P8D scenario uses a 2 GiB CPU transfer buffer per node. Decode-first
requests preserve the full prompt during cache lookup; prefill-first requests
skip the appended first token. The backends stay running for the complete
matrix while a fresh proxy is started for each API and scheduler combination.
For CI-style validation, each combination uses 256 requests at concurrency 128,
for 2,560 requests across the 10 combinations. Override
`GPU_TEST_MATRIX_REQUESTS` and `GPU_TEST_MATRIX_CONCURRENCY` to use a different
load:

```bash
GPU_TEST_MATRIX_REQUESTS=256 \
GPU_TEST_MATRIX_CONCURRENCY=128 \
  bash tests/gpu/run.sh --case lmcache-8p8d --timeout 120
```

`GPU_TEST_LOG_DIR` and `GPU_TEST_TIMEOUT_MINUTES` provide
environment-variable equivalents for the log directory and timeout.

`MODEL=/models/Meta-Llama-3-8B-Instruct` is also supported. The wrapper keeps
the examples' existing logical model names stable while changing only the
weights path passed to vLLM.

For the opt-in GitHub Actions hardware workflow, immutable-SHA result statuses,
runner provisioning and approval requirements, see [the accelerator CI gate](CI.md).

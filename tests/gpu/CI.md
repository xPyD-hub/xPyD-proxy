# Accelerator result gate

`.github/workflows/accelerator.yml` is **manual hardware execution**, separate
from the existing CPU integration dispatcher. Dispatch success is not test
success. The workflow checks out a full, immutable SHA and posts
`accelerator-lifecycle/cuda` or `accelerator-lifecycle/xpu` on that same SHA.
It never resolves a moving branch again at completion.

## Administrator prerequisites

1. Merge the accelerator lifecycle example and this workflow into `main`.
2. Register dedicated Linux self-hosted runners with `xpyd-cuda` or `xpyd-xpu`
   labels, four idle devices, a matching vLLM environment, proxy dependencies,
   `nvidia-smi` or the supported `xpu-smi` memory JSON schema, and local small
   model weights. Jobs run natively in the runner environment; a containerized
   runner must have its accelerator devices mapped in.
3. Create protected environments `accelerator-cuda` and/or `accelerator-xpu`.
   Require maintainer approval, prevent self-approval, restrict deployment
   branches to `main`, and set environment variables
   `ACCELERATOR_MODEL_PATH` and optionally `ACCELERATOR_PYTHON` (executable path).
4. Restrict the runner group to this repository/workflow, prefer ephemeral
   runners, and do not provision cloud credentials or unrelated secrets.
   Approval authorizes execution of the **entire target commit** on hardware;
   review that SHA before approving, especially for contributor PRs.
5. After the first real run creates the status context, an administrator may
   make the applicable platform context a required check in the main ruleset.
   Require both contexts only if both types of runner are available.

This PR does not register machines, create approvals, or modify branch rules.
An absent runner leaves a run queued/pending, not green. Do not enable a
required check before hardware and reviewers are operational.

## Dispatch

In **Actions → Accelerator lifecycle → Run workflow**, choose workflow branch
`main`, enter the reviewed 40-character lowercase SHA, and select `cuda`/`xpu`.
Other workflow branches are intentionally skipped. For a PR, use its current
head SHA; pushing another commit requires a new run. Only a successful hardware
job with all three topology results can create a success status.

The hardware job receives read-only repository access with persisted checkout
credentials disabled. A separate GitHub-hosted job reports status without
checking out target code or consuming executable artifacts. Failures report
failure; cancellation/skipping reports error. If GitHub force-cancels the
reporter or status delivery fails, the gate stays pending rather than claiming
success. Rerunning the workflow resets the target status to pending.

`results.json`, phase-separated scenario/proxy/backend logs, generated configs,
and `tested-sha.txt` are uploaded. The test runs aggregated 2A, direct 2P2D, and
mixed 1P1D+2A with process/port/memory-release checks. Direct mode validates
real accelerator inference and proxy lifecycle, **not NIXL/LMCache KV transfer**.
Those connector matrices need separate compatible hardware coverage.

Different platform runs may execute concurrently; runs for one platform are
serialized and do not cancel active hardware jobs. After cancellation or a
runner crash, confirm that owned processes exited and device memory was
released before accepting another job; ephemeral runner teardown should enforce
that boundary.

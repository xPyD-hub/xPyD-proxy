"""Keep accelerator dispatch separate from execution and privileged reporting."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = yaml.safe_load(
    (
        Path(__file__).resolve().parents[2] / ".github/workflows/accelerator.yml"
    ).read_text()
)


def test_hardware_runs_only_by_manual_approval():
    triggers = WORKFLOW.get("on", WORKFLOW.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    jobs = WORKFLOW["jobs"]
    assert jobs["prepare"]["if"] == "github.ref == 'refs/heads/main'"
    assert jobs["hardware"]["environment"] == "accelerator-${{ inputs.device }}"
    assert jobs["hardware"]["permissions"] == {"contents": "read"}
    assert "self-hosted" in jobs["hardware"]["runs-on"]
    assert WORKFLOW["concurrency"]["cancel-in-progress"] is False


def test_checkout_and_report_use_same_immutable_sha():
    jobs = WORKFLOW["jobs"]
    checkout = jobs["hardware"]["steps"][0]["with"]
    assert checkout["ref"] == "${{ needs.prepare.outputs.sha }}"
    assert checkout["persist-credentials"] is False
    report = jobs["report"]
    assert report["needs"] == ["prepare", "hardware"]
    assert report["if"].startswith("always()")
    assert report["permissions"] == {"statuses": "write"}
    step = report["steps"][0]
    assert step["env"]["TARGET_SHA"] == checkout["ref"]
    assert step["env"]["RESULT"] == "${{ needs.hardware.result }}"
    assert not any("checkout" in item.get("uses", "") for item in report["steps"])


def test_missing_results_and_logs_fail_closed():
    steps = WORKFLOW["jobs"]["hardware"]["steps"]
    result = next(
        step["run"] for step in steps if step.get("name") == "Require complete results"
    )
    compile(result.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0], "<gate>", "exec")
    assert "assert len(results) == 3" in result
    assert 'assert all(item["status"] == "success"' in result
    upload = steps[-1]
    assert upload["if"] == "always()"
    assert upload["with"]["if-no-files-found"] == "error"


def run_script(job, **env):
    if not shutil.which("node"):
        pytest.skip("Node is required to execute GitHub Actions JavaScript")
    script = WORKFLOW["jobs"][job]["steps"][0]["with"]["script"]
    wrapper = (
        """
    const statuses = [];
    let failed = false;
    const context = {repo: {owner: 'owner', repo: 'repo'}, runId: 1,
                     serverUrl: 'https://github.com'};
    const core = {setOutput() {}, setFailed() {failed = true;}};
    const github = {rest: {repos: {
      async getCommit() {return {data: {sha: process.env.RESOLVED_SHA}};},
      async createCommitStatus(status) {statuses.push(status);}
    }}};
    (async () => {
      try {
    """
        + script
        + """
      } catch (error) {failed = true;}
      console.log(JSON.stringify({statuses, failed}));
    })();
    """
    )
    return json.loads(
        subprocess.check_output(
            ["node", "--eval", wrapper],
            text=True,
            env={**os.environ, **env},
        )
    )


@pytest.mark.parametrize(
    "result,state",
    [
        ("success", "success"),
        ("failure", "failure"),
        ("cancelled", "error"),
        ("skipped", "error"),
        ("unknown", "error"),
    ],
)
def test_report_never_treats_skipped_or_cancelled_as_success(result, state):
    output = run_script("report", TARGET_SHA="a" * 40, DEVICE="xpu", RESULT=result)
    assert output["failed"] == (state != "success")
    assert len(output["statuses"]) == 1
    assert output["statuses"][0]["sha"] == "a" * 40
    assert output["statuses"][0]["state"] == state


@pytest.mark.parametrize(
    "sha,resolved,valid",
    [
        ("a" * 40, "a" * 40, True),
        ("a" * 40, "b" * 40, False),
        ("main", "a" * 40, False),
        ("a" * 7, "a" * 40, False),
    ],
)
def test_prepare_requires_exact_sha(sha, resolved, valid):
    output = run_script("prepare", TARGET_SHA=sha, RESOLVED_SHA=resolved, DEVICE="cuda")
    assert output["failed"] != valid
    assert len(output["statuses"]) == int(valid)
    if valid:
        assert output["statuses"][0]["state"] == "pending"

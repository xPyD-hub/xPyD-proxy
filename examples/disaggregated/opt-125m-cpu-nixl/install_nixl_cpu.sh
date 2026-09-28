#!/usr/bin/env bash

set -euo pipefail

NIXL_VERSION="${NIXL_VERSION:-v1.3.0}"
VLLM_VERSION="${VLLM_VERSION:-0.25.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ID="$(python "${SCRIPT_DIR}/wheel_cache.py" fingerprint \
    --nixl "${NIXL_VERSION}" --vllm "${VLLM_VERSION}")"
export WHEELS_CACHE_HOME="${WHEELS_CACHE_HOME:-${HOME}/.cache/xpyd-nixl-wheels}/${CACHE_ID}"
# This example runs every P/D process on one host. Loopback avoids depending
# on cloud-runner NIC metadata; multi-host users can override this variable.
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-lo}"

mkdir -p "${WHEELS_CACHE_HOME}"
CACHED_WHEEL="$(python "${SCRIPT_DIR}/wheel_cache.py" select \
    --nixl "${NIXL_VERSION}" --directory "${WHEELS_CACHE_HOME}")"

if [[ -n "${CACHED_WHEEL}" ]]; then
    echo "Installing fingerprinted CPU NIXL wheel: ${CACHED_WHEEL}"
    python -m pip install --force-reinstall --no-deps "${CACHED_WHEEL}"
else
    sudo apt-get update
    sudo apt-get install -y \
        automake \
        autotools-dev \
        build-essential \
        cmake \
        libtool \
        libtool-bin \
        liburing-dev \
        meson \
        ninja-build \
        patchelf \
        pkg-config

installer="$(mktemp)"
trap 'rm -f "${installer}"' EXIT

curl --fail --location --retry 3 \
    "https://raw.githubusercontent.com/vllm-project/vllm/v${VLLM_VERSION}/tools/install_nixl_from_source_ubuntu.py" \
    --output "${installer}"

# Ubuntu 22.04 provides patchelf 0.14.3, while current auditwheel requires
# at least 0.14.5. The venv binary takes precedence over the apt package.
python -m pip install "patchelf>=0.14.5"

# Build only the transport used by this example. The default plugin set also
# builds POSIX support and leaves auditwheel with an unavailable liburing.so.2.
sed -i \
    '/f"--wheel-dir={temp_wheel_dir}",/a\            "--config-settings=setup-args=-Denable_plugins=UCX",' \
    "${installer}"

python - "${installer}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
content = path.read_text()
old_pattern = 'f"nixl*{NIXL_VERSION}*.whl"'
if old_pattern not in content:
    raise RuntimeError("Upstream wheel lookup changed; review the installer patch")
content = content.replace(
    old_pattern, 'f"nixl*-{NIXL_VERSION.removeprefix(\'v\')}-*.whl"', 1
)
content = content.replace(
    "import subprocess\n",
    "import subprocess\nimport shutil\nimport tempfile\nimport zipfile\n",
    1,
)
content = content.replace(
    "    auditwheel_command = [\n",
    """    wheel_extract_dir = tempfile.mkdtemp(prefix="nixl-wheel-")
    with zipfile.ZipFile(unrepaired_wheel) as wheel:
        wheel.extractall(wheel_extract_dir)
    internal_libraries = glob.glob(
        os.path.join(
            wheel_extract_dir,
            ".*.mesonpy.libs",
            "**",
            "*.so*",
        ),
        recursive=True,
    )
    internal_lib_dirs = sorted({
        os.path.dirname(library)
        for library in internal_libraries
    })
    if not internal_lib_dirs:
        raise RuntimeError("NIXL wheel did not contain internal shared libraries")
    for library in internal_libraries:
        soname = subprocess.check_output(
            ["patchelf", "--print-soname", library],
            text=True,
        ).strip()
        soname_path = os.path.join(os.path.dirname(library), soname)
        if soname and not os.path.exists(soname_path):
            os.symlink(os.path.basename(library), soname_path)
    build_env["LD_LIBRARY_PATH"] = ":".join(
        internal_lib_dirs + [build_env.get("LD_LIBRARY_PATH", "")]
    ).strip(":")

    auditwheel_command = [
""",
    1,
)
content = content.replace(
    "    run_command(auditwheel_command, env=build_env)\n",
    """    run_command(auditwheel_command, env=build_env)
    shutil.rmtree(wheel_extract_dir)
""",
    1,
)
path.write_text(content)
PY

# vLLM may already have installed a different NIXL. Do not let the upstream
# installer's package-presence shortcut skip the requested CPU build.
NIXL_VERSION="${NIXL_VERSION}" python "${installer}" --force-reinstall
python "${SCRIPT_DIR}/wheel_cache.py" select \
    --nixl "${NIXL_VERSION}" --directory "${WHEELS_CACHE_HOME}"
fi
platform_version="$(
    python -c 'import importlib.metadata as m; print(m.version("nixl-cu12"))'
)"
[[ "${platform_version}" == "${NIXL_VERSION#v}" ]] || {
    echo "ERROR: expected NIXL ${NIXL_VERSION#v}, installed ${platform_version}" >&2
    exit 1
}
python -m pip install --force-reinstall --no-deps "nixl==${platform_version}"
UCX_TLS=tcp python - <<'PY'
import os
from pathlib import Path
import tempfile

import nixl

with tempfile.TemporaryDirectory(prefix="xpyd-nixl-telemetry-") as telemetry_dir:
    os.environ["NIXL_TELEMETRY_ENABLE"] = "y"
    os.environ["NIXL_TELEMETRY_DIR"] = telemetry_dir
    agent = nixl.nixl_agent("xpyd-cpu-check")
    assert agent is not None
    assert (Path(telemetry_dir) / "xpyd-cpu-check").is_file()
    del agent

print("NIXL CPU/UCX and telemetry initialization passed.")
PY

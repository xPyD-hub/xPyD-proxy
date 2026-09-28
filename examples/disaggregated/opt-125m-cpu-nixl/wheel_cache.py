#!/usr/bin/env python3
"""Fingerprint the CPU build inputs and reject incompatible cached wheels."""

import argparse
import hashlib
import json
import platform
import sysconfig
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version


def fingerprint(nixl, vllm):
    root = Path(__file__).resolve().parent
    inputs = {
        "nixl": str(Version(nixl.removeprefix("v"))),
        "vllm_installer": vllm,
        "platform": (platform.system(), platform.machine()),
        "distribution": platform.freedesktop_os_release(),
        "libc": platform.libc_ver(),
        "abi": sysconfig.get_config_var("SOABI"),
        "recipe": hashlib.sha256(
            (root / "install_nixl_cpu.sh").read_bytes() + Path(__file__).read_bytes()
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:24]


def cached_wheel(directory, version):
    compatible = set(sys_tags())
    expected = Version(version.removeprefix("v"))
    candidates = []
    for path in sorted(directory.glob("*.whl")):
        name, actual, _, tags = parse_wheel_filename(path.name)
        if name != "nixl-cu12" or actual != expected or not tags & compatible:
            raise ValueError(f"Incompatible wheel in fingerprinted cache: {path.name}")
        candidates.append(path)
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous NIXL wheel cache: {directory}")
    return str(candidates[0]) if candidates else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fingerprint", "select"))
    parser.add_argument("--nixl", required=True)
    parser.add_argument("--vllm", default="0.25.0")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.command == "fingerprint":
        print(fingerprint(args.nixl, args.vllm))
    else:
        if args.directory is None:
            parser.error("select requires --directory")
        print(cached_wheel(args.directory, args.nixl))


if __name__ == "__main__":
    main()

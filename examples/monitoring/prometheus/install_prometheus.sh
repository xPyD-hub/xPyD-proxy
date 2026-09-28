#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION=3.5.0
case "$(uname -m)" in
    x86_64) ARCH=amd64 ;;
    aarch64) ARCH=arm64 ;;
    *) echo "ERROR: this example supports Linux amd64/arm64." >&2; exit 1 ;;
esac
[[ "$(uname -s)" == Linux ]] || {
    echo "ERROR: this installer supports Linux only." >&2
    exit 1
}
NAME="prometheus-${VERSION}.linux-${ARCH}"
DEST="${SCRIPT_DIR}/logs/tools"
mkdir -p "${DEST}"
if [[ -x "${DEST}/${NAME}/prometheus" && -x "${DEST}/${NAME}/promtool" ]]; then
    printf '%s\n' "${DEST}/${NAME}"
    exit 0
fi
TMP="$(mktemp -d "${DEST}/download.XXXXXX")"
trap 'rm -f "${TMP}/archive.tar.gz" "${TMP}/sha256sums.txt"; rmdir "${TMP}"' EXIT
BASE="https://github.com/prometheus/prometheus/releases/download/v${VERSION}"
curl --fail --location --retry 3 --connect-timeout 15 --max-time 300 \
    "${BASE}/${NAME}.tar.gz" -o "${TMP}/archive.tar.gz" >&2
curl --fail --location --retry 3 --connect-timeout 15 --max-time 120 \
    "${BASE}/sha256sums.txt" -o "${TMP}/sha256sums.txt" >&2
CHECKSUM="$(awk -v name="${NAME}.tar.gz" '$2 == name {print $1}' "${TMP}/sha256sums.txt")"
[[ "${CHECKSUM}" =~ ^[a-f0-9]{64}$ ]] || {
    echo "ERROR: release checksum not found." >&2
    exit 1
}
printf '%s  %s\n' "${CHECKSUM}" "${TMP}/archive.tar.gz" | sha256sum --check >&2
tar -xzf "${TMP}/archive.tar.gz" -C "${DEST}" \
    "${NAME}/prometheus" "${NAME}/promtool"
printf '%s\n' "${DEST}/${NAME}"

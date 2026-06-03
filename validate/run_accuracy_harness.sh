#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ttmlir_root="${TTMLIR_ROOT:-$repo_root/../tt-mlir}"

if [[ -f "$ttmlir_root/env/activate" ]]; then
  # shellcheck disable=SC1090
  set +u
  source "$ttmlir_root/env/activate"
  set -u
fi

exec python3 "$repo_root/validate/accuracy_harness.py" "$@"

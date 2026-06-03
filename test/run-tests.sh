#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build"
maps_opt="${build_dir}/tools/maps-opt/maps-opt"
filecheck="${FILECHECK:-/opt/ttmlir-toolchain/bin/FileCheck}"
ttmlir_opt="${TTMLIR_OPT:-/home/ivan/repos/tt-mlir/build/bin/ttmlir-opt}"

cmake --build "${build_dir}" --target maps-opt -j2 >/dev/null

if [[ ! -x "${ttmlir_opt}" ]]; then
  echo "missing ttmlir-opt: ${ttmlir_opt}" >&2
  exit 1
fi

verify_downstream_pipeline() {
  local test_file="$1"

  echo "[pipeline] ${test_file}"
  "${maps_opt}" "${test_file}" -convert-maps-to-d2m \
    | "${ttmlir_opt}" --ttcore-register-device --d2m-fe-pipeline \
    | "${ttmlir_opt}" --ttcore-register-device --d2m-be-pipeline -o /dev/null
}

run_file() {
  local test_file="$1"
  local run_line
  run_line="$(grep -m1 '^// RUN:' "${test_file}" | sed 's#^// RUN: ##')"
  if [[ -z "${run_line}" ]]; then
    echo "missing RUN line: ${test_file}" >&2
    return 1
  fi

  local command="${run_line//maps-opt/${maps_opt}}"
  command="${command//FileCheck/${filecheck}}"
  command="${command//%s/${test_file}}"

  echo "[test] ${test_file}"
  bash -lc "${command}"

  if [[ "${run_line}" == *"-convert-maps-to-d2m"* ]]; then
    verify_downstream_pipeline "${test_file}"
  fi
}

shopt -s nullglob
for test_file in "${repo_root}"/test/Conversion/MapsToD2M/*.mlir; do
  run_file "${test_file}"
done

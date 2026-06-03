#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from accuracy_references import REFERENCE_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "validate" / "accuracy_corpus.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "generated" / "accuracy_harness"
DEFAULT_TTMLIR_ROOT = (REPO_ROOT / ".." / "tt-mlir").resolve()
DEFAULT_TTSIM_ROOT = (REPO_ROOT / ".." / "ttsim").resolve()


@dataclass
class CommandResult:
    returncode: int
    stdout_path: Path
    stderr_path: Path


def require_module(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    raise SystemExit(
        f"missing Python module '{module_name}'. Run the harness via "
        f"`validate/run_accuracy_harness.sh` so the tt-mlir environment is active."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MAPS accuracy harness on a checked-in corpus."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the corpus manifest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory used for per-case artifacts and summaries.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_filters",
        help="Run only the named case. Can be passed multiple times.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the manifest seed for all cases.",
    )
    parser.add_argument(
        "--ttmlir-root",
        type=Path,
        default=DEFAULT_TTMLIR_ROOT,
        help="Adjacent tt-mlir checkout root.",
    )
    parser.add_argument(
        "--ttsim-root",
        type=Path,
        default=DEFAULT_TTSIM_ROOT,
        help="Adjacent ttsim checkout root.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "cases" not in manifest or not isinstance(manifest["cases"], list):
        raise SystemExit(f"invalid manifest: {path}")
    return manifest


def filtered_cases(
    manifest: Dict[str, Any], case_filters: Optional[Sequence[str]]
) -> List[Dict[str, Any]]:
    cases = manifest["cases"]
    if not case_filters:
        return cases
    selected = [case for case in cases if case["name"] in set(case_filters)]
    missing = sorted(set(case_filters) - {case["name"] for case in selected})
    if missing:
        raise SystemExit(f"unknown case filter(s): {', '.join(missing)}")
    return selected


def ensure_tool(path: Path, label: str) -> None:
    if path.is_file() and os.access(path, os.X_OK):
        return
    raise SystemExit(f"missing {label}: {path}")


def resolve_ttrt_command(env: Dict[str, str], ttmlir_root: Path) -> List[str]:
    if shutil.which("ttrt"):
        return ["ttrt"]
    if importlib.util.find_spec("ttrt") is not None:
        return [sys.executable, "-m", "ttrt"]
    local_ttrt = ttmlir_root / "build" / "tools" / "ttrt"
    if local_ttrt.is_dir():
        env["PYTHONPATH"] = (
            str(local_ttrt)
            if not env.get("PYTHONPATH")
            else f"{local_ttrt}:{env['PYTHONPATH']}"
        )
        return [sys.executable, "-m", "ttrt"]
    raise SystemExit("could not find a usable ttrt command")


def resolve_simulator(env: Dict[str, str], ttsim_root: Path) -> None:
    current = env.get("TT_METAL_SIMULATOR")
    if current and Path(current).is_file():
        return
    candidates = [
        ttsim_root / "libttsim_wh.so",
        ttsim_root / "build",
        ttsim_root / "lib",
        ttsim_root,
    ]
    for candidate in candidates:
        if candidate.is_file():
            env["TT_METAL_SIMULATOR"] = str(candidate)
            break
        if candidate.is_dir():
            matches = sorted(candidate.glob("libttsim*.so"))
            if matches:
                env["TT_METAL_SIMULATOR"] = str(matches[0])
                break
    if env.get("TT_METAL_SIMULATOR"):
        env.setdefault("TT_METAL_DISABLE_SFPLOADMACRO", "1")
        env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")


def run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> CommandResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    return CommandResult(
        returncode=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def copy_or_translate_source(
    case: Dict[str, Any],
    *,
    maps_translate: Path,
    env: Dict[str, str],
    case_dir: Path,
) -> Path:
    source_path = (REPO_ROOT / case["source"]).resolve()
    output_path = case_dir / "ingest" / "maps_input.mlir"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_format = case["source_format"]
    if source_format == "maps_json":
        result = run_command(
            [str(maps_translate), "--json-to-maps", str(source_path), "-o", str(output_path)],
            cwd=REPO_ROOT,
            env=env,
            stdout_path=case_dir / "logs" / "ingest.stdout.log",
            stderr_path=case_dir / "logs" / "ingest.stderr.log",
        )
        if result.returncode != 0:
            raise RuntimeError("ingest failure")
    elif source_format == "maps_mlir":
        shutil.copyfile(source_path, output_path)
        (case_dir / "logs").mkdir(parents=True, exist_ok=True)
        (case_dir / "logs" / "ingest.stdout.log").write_text("", encoding="utf-8")
        (case_dir / "logs" / "ingest.stderr.log").write_text("", encoding="utf-8")
    else:
        raise RuntimeError(f"unsupported source_format: {source_format}")
    return output_path


def lower_to_d2m(
    input_mlir: Path,
    *,
    maps_opt: Path,
    env: Dict[str, str],
    case_dir: Path,
) -> Path:
    output_path = case_dir / "lowering" / "lowered_d2m.mlir"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [str(maps_opt), str(input_mlir), "-convert-maps-to-d2m", "-o", str(output_path)],
        cwd=REPO_ROOT,
        env=env,
        stdout_path=case_dir / "logs" / "lowering.stdout.log",
        stderr_path=case_dir / "logs" / "lowering.stderr.log",
    )
    if result.returncode != 0:
        raise RuntimeError("MAPS lowering failure")
    return output_path


def compile_backend(
    input_d2m: Path,
    *,
    emit_exec: Path,
    env: Dict[str, str],
    case_dir: Path,
) -> Path:
    output_dir = case_dir / "compile"
    result = run_command(
        [str(emit_exec), str(input_d2m), str(output_dir)],
        cwd=REPO_ROOT,
        env=env,
        stdout_path=case_dir / "logs" / "backend_compile.stdout.log",
        stderr_path=case_dir / "logs" / "backend_compile.stderr.log",
    )
    if result.returncode != 0:
        raise RuntimeError("backend compile failure")
    executable = output_dir / "140_executable.ttm"
    if not executable.is_file():
        raise RuntimeError("backend compile failure")
    return executable


def runtime_run(
    executable: Path,
    *,
    ttrt_command: Sequence[str],
    env: Dict[str, str],
    case_dir: Path,
    seed: int,
    init: str,
) -> Tuple[Path, Path]:
    runtime_dir = case_dir / "runtime"
    artifact_dir = runtime_dir / "artifacts"
    result_file = runtime_dir / "run_results.json"
    log_file = runtime_dir / "ttrt.log"
    result = run_command(
        [
            *ttrt_command,
            "run",
            str(executable),
            "--clean-artifacts",
            "--save-artifacts",
            "--artifact-dir",
            str(artifact_dir),
            "--log-file",
            str(log_file),
            "--result-file",
            str(result_file),
            "--seed",
            str(seed),
            "--init",
            init,
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout_path=case_dir / "logs" / "runtime.stdout.log",
        stderr_path=case_dir / "logs" / "runtime.stderr.log",
    )
    if result.returncode != 0:
        raise RuntimeError("runtime execution failure")
    return artifact_dir, result_file


def load_runtime_tensors(
    artifact_dir: Path, executable: Path, *, torch_module: Any
) -> Tuple[List[Any], List[Any]]:
    program_dir = artifact_dir / executable.name / "run" / "program_0"
    if not program_dir.is_dir():
        raise RuntimeError(f"missing runtime artifact directory: {program_dir}")
    input_paths = sorted(program_dir.glob("input_*.pt"))
    output_paths = sorted(program_dir.glob("device_output_*.pt"))
    inputs = [
        torch_module.load(path, weights_only=True).detach().cpu() for path in input_paths
    ]
    outputs = [
        torch_module.load(path, weights_only=True).detach().cpu() for path in output_paths
    ]
    return inputs, outputs


def tensor_dtype_name(tensor: Any) -> str:
    return str(tensor.dtype).replace("torch.", "")


def validate_tensor_specs(
    tensors: Sequence[Any], specs: Sequence[Dict[str, Any]], *, kind: str
) -> None:
    if len(tensors) != len(specs):
        raise RuntimeError(f"{kind} count mismatch: expected {len(specs)}, got {len(tensors)}")
    for index, (tensor, spec) in enumerate(zip(tensors, specs)):
        expected_shape = tuple(spec["shape"])
        actual_shape = tuple(int(dim) for dim in tensor.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{kind} {index} shape mismatch: expected {expected_shape}, got {actual_shape}"
            )
        expected_dtype = spec.get("dtype")
        actual_dtype = tensor_dtype_name(tensor)
        if expected_dtype and actual_dtype != expected_dtype:
            raise RuntimeError(
                f"{kind} {index} dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
            )


def compute_pcc(actual: Any, expected: Any, *, torch_module: Any) -> float:
    actual_flat = actual.detach().to(torch_module.float32).reshape(-1)
    expected_flat = expected.detach().to(torch_module.float32).reshape(-1)
    if actual_flat.numel() != expected_flat.numel():
        raise RuntimeError("tensor size mismatch during PCC calculation")
    if actual_flat.numel() == 1:
        if float(expected_flat.item()) == 0.0:
            return 1.0 if float(actual_flat.item()) == 0.0 else 0.0
        return float(
            torch_module.nn.functional.cosine_similarity(
                expected_flat.unsqueeze(0), actual_flat.unsqueeze(0)
            ).item()
        )
    expected_centered = expected_flat - expected_flat.mean()
    actual_centered = actual_flat - actual_flat.mean()
    expected_norm = float(expected_centered.norm().item())
    actual_norm = float(actual_centered.norm().item())
    if expected_norm == 0.0 or actual_norm == 0.0:
        return 1.0 if torch_module.equal(actual_flat, expected_flat) else 0.0
    return float(
        torch_module.dot(expected_centered, actual_centered).item()
        / (expected_norm * actual_norm)
    )


def compare_outputs(
    case: Dict[str, Any],
    runtime_inputs: List[Any],
    runtime_outputs: List[Any],
    defaults: Dict[str, Any],
    *,
    torch_module: Any,
) -> Dict[str, Any]:
    validate_tensor_specs(runtime_inputs, case["inputs"], kind="input")
    validate_tensor_specs(runtime_outputs, case["outputs"], kind="output")
    reference_name = case["reference"]
    reference_fn = REFERENCE_REGISTRY.get(reference_name)
    if reference_fn is None:
        raise RuntimeError(f"unknown reference entrypoint: {reference_name}")
    expected_outputs = reference_fn(runtime_inputs)
    if len(expected_outputs) != len(runtime_outputs):
        raise RuntimeError(
            f"reference output count mismatch: expected {len(runtime_outputs)}, got {len(expected_outputs)}"
        )

    per_output = []
    all_passed = True
    for index, (actual, expected, output_spec) in enumerate(
        zip(runtime_outputs, expected_outputs, case["outputs"])
    ):
        atol = float(output_spec.get("atol", case.get("atol", defaults["atol"])))
        rtol = float(output_spec.get("rtol", case.get("rtol", defaults["rtol"])))
        pcc_threshold = float(output_spec.get("pcc", case.get("pcc", defaults["pcc"])))
        actual_cpu = actual.detach().cpu()
        expected_cpu = expected.detach().cpu().to(actual_cpu.dtype)
        pcc = compute_pcc(actual_cpu, expected_cpu, torch_module=torch_module)
        allclose = bool(
            torch_module.allclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol, equal_nan=True)
        )
        max_abs_error = float(
            torch_module.max(
                torch_module.abs(actual_cpu.to(torch_module.float32) - expected_cpu.to(torch_module.float32))
            ).item()
        )
        passed = allclose and pcc >= pcc_threshold
        all_passed = all_passed and passed
        per_output.append(
            {
                "index": index,
                "name": output_spec.get("name", f"output_{index}"),
                "dtype": tensor_dtype_name(actual_cpu),
                "shape": [int(dim) for dim in actual_cpu.shape],
                "pass": passed,
                "pcc": pcc,
                "pcc_threshold": pcc_threshold,
                "atol": atol,
                "rtol": rtol,
                "allclose": allclose,
                "max_abs_error": max_abs_error,
            }
        )
    return {"pass": all_passed, "outputs": per_output}


def summarize_case(
    case: Dict[str, Any],
    *,
    case_dir: Path,
    seed: int,
    init: str,
) -> Dict[str, Any]:
    return {
        "name": case["name"],
        "description": case.get("description", ""),
        "source": case["source"],
        "case_dir": str(case_dir),
        "seed": seed,
        "init": init,
        "compile_pass": False,
        "run_pass": False,
        "numeric_pass": False,
        "failure_stage": None,
        "stages": {},
        "metrics": {"outputs": []},
    }


def mark_stage(summary: Dict[str, Any], stage: str, status: str, **extra: Any) -> None:
    stage_entry = {"status": status, **extra}
    summary["stages"][stage] = stage_entry
    if status != "passed" and summary["failure_stage"] is None:
        summary["failure_stage"] = stage


def stage_log_paths(case_dir: Path, stage: str) -> Dict[str, str]:
    return {
        "stdout_log": str(case_dir / "logs" / f"{stage}.stdout.log"),
        "stderr_log": str(case_dir / "logs" / f"{stage}.stderr.log"),
    }


def print_table(results: List[Dict[str, Any]]) -> None:
    headers = ("case", "compile", "run", "numeric", "failure_stage", "max_abs_error", "min_pcc")
    rows = []
    for result in results:
        output_metrics = result["metrics"]["outputs"]
        max_abs_error = "-"
        min_pcc = "-"
        if output_metrics:
            max_abs_error = f"{max(metric['max_abs_error'] for metric in output_metrics):.4e}"
            min_pcc = f"{min(metric['pcc'] for metric in output_metrics):.6f}"
        rows.append(
            (
                result["name"],
                "pass" if result["compile_pass"] else "fail",
                "pass" if result["run_pass"] else "fail",
                "pass" if result["numeric_pass"] else "fail",
                result["failure_stage"] or "-",
                max_abs_error,
                min_pcc,
            )
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    format_row = "  ".join(f"{{:{width}}}" for width in widths)
    print(format_row.format(*headers))
    print(format_row.format(*("-" * width for width in widths)))
    for row in rows:
        print(format_row.format(*row))


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total_cases": len(results),
        "compile_pass_count": sum(1 for result in results if result["compile_pass"]),
        "run_pass_count": sum(1 for result in results if result["run_pass"]),
        "numeric_pass_count": sum(1 for result in results if result["numeric_pass"]),
        "cases": results,
    }


def main() -> int:
    require_module("torch")
    import torch

    args = parse_args()
    manifest = load_manifest(args.manifest)
    defaults = dict(manifest.get("defaults", {}))
    defaults.setdefault("seed", 0)
    defaults.setdefault("init", "randn")
    defaults.setdefault("pcc", 0.99)
    defaults.setdefault("atol", 1e-3)
    defaults.setdefault("rtol", 5e-2)

    cases = filtered_cases(manifest, args.case_filters)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    maps_opt = REPO_ROOT / "build" / "tools" / "maps-opt" / "maps-opt"
    maps_translate = REPO_ROOT / "build" / "tools" / "maps-translate" / "maps-translate"
    emit_exec = REPO_ROOT / "dev" / "emit_exec.sh"
    ensure_tool(maps_opt, "maps-opt")
    ensure_tool(maps_translate, "maps-translate")
    ensure_tool(emit_exec, "dev/emit_exec.sh")

    env = os.environ.copy()
    env.setdefault("TTMLIR_ROOT", str(args.ttmlir_root.resolve()))
    env.setdefault("TTSIM_ROOT", str(args.ttsim_root.resolve()))
    env.setdefault("TTRT_IGNORE_VERSION", "0")
    cache_root = output_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    env.setdefault("XDG_CACHE_HOME", str(cache_root))
    env.setdefault("TT_METAL_CACHE", str(cache_root / "tt-metal-cache"))
    Path(env["TT_METAL_CACHE"]).mkdir(parents=True, exist_ok=True)
    resolve_simulator(env, Path(env["TTSIM_ROOT"]))
    ttrt_command = resolve_ttrt_command(env, Path(env["TTMLIR_ROOT"]))

    results: List[Dict[str, Any]] = []
    for case in cases:
        seed = args.seed if args.seed is not None else int(case.get("seed", defaults["seed"]))
        init = str(case.get("init", defaults["init"]))
        case_dir = output_root / case["name"]
        summary = summarize_case(case, case_dir=case_dir, seed=seed, init=init)
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            ingested_mlir = copy_or_translate_source(
                case, maps_translate=maps_translate, env=env, case_dir=case_dir
            )
            mark_stage(
                summary,
                "ingest",
                "passed",
                artifact=str(ingested_mlir),
                **stage_log_paths(case_dir, "ingest"),
            )

            d2m_mlir = lower_to_d2m(ingested_mlir, maps_opt=maps_opt, env=env, case_dir=case_dir)
            mark_stage(
                summary,
                "maps_lowering",
                "passed",
                artifact=str(d2m_mlir),
                **stage_log_paths(case_dir, "lowering"),
            )

            executable = compile_backend(d2m_mlir, emit_exec=emit_exec, env=env, case_dir=case_dir)
            summary["compile_pass"] = True
            mark_stage(
                summary,
                "backend_compile",
                "passed",
                artifact=str(executable),
                **stage_log_paths(case_dir, "backend_compile"),
            )

            artifact_dir, result_file = runtime_run(
                executable,
                ttrt_command=ttrt_command,
                env=env,
                case_dir=case_dir,
                seed=seed,
                init=init,
            )
            summary["run_pass"] = True
            mark_stage(
                summary,
                "runtime_execution",
                "passed",
                artifact_dir=str(artifact_dir),
                result_file=str(result_file),
                **stage_log_paths(case_dir, "runtime"),
            )

            runtime_inputs, runtime_outputs = load_runtime_tensors(
                artifact_dir, executable, torch_module=torch
            )
            metrics = compare_outputs(
                case,
                runtime_inputs,
                runtime_outputs,
                defaults,
                torch_module=torch,
            )
            summary["metrics"] = metrics
            summary["numeric_pass"] = bool(metrics["pass"])
            mark_stage(
                summary,
                "numeric_compare",
                "passed" if metrics["pass"] else "failed",
            )
        except RuntimeError as error:
            message = str(error)
            if "ingest failure" in message:
                mark_stage(
                    summary,
                    "ingest",
                    "failed",
                    message=message,
                    **stage_log_paths(case_dir, "ingest"),
                )
            elif "MAPS lowering failure" in message:
                mark_stage(
                    summary,
                    "maps_lowering",
                    "failed",
                    message=message,
                    **stage_log_paths(case_dir, "lowering"),
                )
            elif "backend compile failure" in message:
                mark_stage(
                    summary,
                    "backend_compile",
                    "failed",
                    message=message,
                    **stage_log_paths(case_dir, "backend_compile"),
                )
            elif "runtime execution failure" in message:
                mark_stage(
                    summary,
                    "runtime_execution",
                    "failed",
                    message=message,
                    **stage_log_paths(case_dir, "runtime"),
                )
            else:
                mark_stage(summary, "numeric_compare", "failed", message=message)
        results.append(summary)

    final_summary = build_summary(results)
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)

    print(f"Summary JSON: {summary_path}")
    print_table(results)
    return 0 if final_summary["numeric_pass_count"] == final_summary["total_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

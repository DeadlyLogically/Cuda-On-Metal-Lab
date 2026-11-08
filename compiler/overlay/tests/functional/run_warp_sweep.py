#!/usr/bin/env python3

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from run_warp_launcher import PIN, provenance, sha256

CHILD = r'''
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import warp as wp

identifier, selected_device, report_path = sys.argv[1:]
def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(wp.config.cuda_output is None, "Application must not select PTX")
for key, value in {
    "CUMETAL_DRIVER_JIT": "runtime-msl", "CUMETAL_PTX_BACKEND": "cumetal-ir",
    "CUMETAL_FP64_MODE": "ieee64", "CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS": "0",
    "CUMETAL_ENABLE_LLMC_CPU_EMULATION": "0", "CUMETAL_DISABLE_LLMC_EMULATION": "1",
}.items():
    require(os.environ.get(key) == value, f"Unexpected {key}")
wp.init()
device = wp.get_device(selected_device)
if selected_device == "cuda:0":
    require(device.is_cuda and device.name.startswith("Apple "), "Expected Apple GPU")
    require(device.get_cuda_output_format() == "ptx", "Automatic PTX selection failed")
else:
    require(device.is_cpu, "Expected CPU baseline")

module_name = identifier.rsplit(".", 2)[0]
module = importlib.import_module(module_name)
checkout = Path(wp.__file__).resolve().parent.parent
source = Path(module.__file__).resolve()
relative = source.relative_to(checkout).as_posix()
original = subprocess.check_output(["git", "-C", str(checkout), "show", "HEAD:" + relative])
require(source.read_bytes() == original, "Original test/calculations were modified")
suite = unittest.defaultTestLoader.loadTestsFromName(identifier)
require(suite.countTestCases() == 1, "Manifest must select exactly one original test")
result = unittest.TextTestRunner(verbosity=2).run(suite)
source_unchanged = source.read_bytes() == original
passed = (result.wasSuccessful() and result.testsRun == 1 and not result.skipped
          and not result.expectedFailures and not result.unexpectedSuccesses
          and source_unchanged and wp.config.cuda_output is None)
report = {
    "passed": passed, "test": identifier, "tests_run": result.testsRun,
    "failures": len(result.failures), "errors": len(result.errors),
    "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
    "unexpected_successes": len(result.unexpectedSuccesses),
    "failure_details": [{"test": case.id(), "traceback": detail} for case, detail in result.failures],
    "error_details": [{"test": case.id(), "traceback": detail} for case, detail in result.errors],
    "skip_details": [{"test": case.id(), "reason": reason} for case, reason in result.skipped],
    "device": selected_device, "device_name": device.name,
    "output_config": wp.config.cuda_output, "selected_output": device.get_cuda_output_format(),
    "warp_version": wp.__version__, "original_source": relative,
    "source_sha256": hashlib.sha256(original).hexdigest(), "source_unchanged": source_unchanged,
}
Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(0 if passed else 1)
'''

def load_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest.get("warp_commit") != PIN or not isinstance(manifest.get("cases"), list):
        raise ValueError("manifest must contain the pinned Warp commit and a cases list")
    names, identifiers = set(), set()
    for case in manifest["cases"]:
        name, identifier = case.get("case", ""), case.get("test", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise ValueError(f"invalid or duplicate case name: {name!r}")
        if not re.fullmatch(r"warp\.tests\.[A-Za-z0-9_.]+\.test_[A-Za-z0-9_]+_cuda_0", identifier):
            raise ValueError(f"invalid original GPU test identifier: {identifier!r}")
        if identifier in identifiers or not case.get("category"):
            raise ValueError(f"duplicate identifier or missing category: {name}")
        names.add(name)
        identifiers.add(identifier)
    if not names:
        raise ValueError("manifest cannot be empty")
    return manifest

def validate_result(checks, records, code, timed_out, device, identifier):
    errors = []
    if timed_out:
        errors.append("process timed out")
    if code != 0:
        errors.append(f"process exit {code}")
    numerical = bool(checks and checks.get("passed") is True
        and checks.get("test") == identifier and checks.get("tests_run") == 1
        and all(checks.get(key) == 0 for key in
                ("failures", "errors", "skipped", "expected_failures", "unexpected_successes"))
        and checks.get("source_unchanged") is True and checks.get("output_config") is None
        and checks.get("device") == device)
    if not numerical:
        errors.append("original numerical test did not pass without skips or source changes")
    if device == "cuda:0":
        if not checks or checks.get("selected_output") != "ptx" or not checks.get("device_name", "").startswith("Apple "):
            errors.append("automatic PTX/Apple GPU selection was not established")
        if not records:
            errors.append("no GPU launches observed")
        allowed = {"generic_ptx_lowering", "generic_ptx_lowering_fp64_ieee64"}
        invalid = [record for record in records if not (
            record.get("event") == "kernel_launch" and record.get("source") == "generic_ptx"
            and record.get("provenance") in allowed and record.get("device") == "apple_gpu"
            and record.get("device_name") == (checks or {}).get("device_name")
            and record.get("launch_success") == "true" and record.get("semantic_quality") == "exact")]
        if invalid:
            errors.append("non-generic, failed, approximate, CPU, or unexpected-device GPU trace")
    elif records:
        errors.append("CPU baseline unexpectedly launched GPU work")
    return errors

def failure_kind(checks, code, timed_out, errors):
    if not errors:
        return "pass"
    if timed_out:
        return "timeout"
    if code < 0:
        return "crash"
    if checks and checks.get("skipped", 0):
        return "skip"
    if checks and any(checks.get(key, 0) for key in ("failures", "errors", "expected_failures", "unexpected_successes")):
        return "test_failure"
    if code != 0 or not checks:
        return "process_failure"
    return "evidence_failure"

def finalize_report(report, identity_after):
    report["complete"] = True
    report["identity_after"] = identity_after
    report["artifacts_unchanged"] = report["identity"] == identity_after
    report["run_errors"] = [] if report["artifacts_unchanged"] else ["compiler/runtime artifacts changed during the run"]
    report["passed"] = (len(report["cases"]) == report["selected_total"]
                        and all(case["passed"] for case in report["cases"])
                        and report["artifacts_unchanged"])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cumetal", "warp", "python", "clang", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("warp_sweep_manifest.json"))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--save-temps", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 180:
        parser.error("timeout must be between 1 and 180 seconds")
    try:
        manifest = load_manifest(args.manifest)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    known = {case["case"] for case in manifest["cases"]}
    if set(args.case) - known:
        parser.error("unknown cases: " + ", ".join(sorted(set(args.case) - known)))
    cases = [case for case in manifest["cases"] if not args.case or case["case"] in args.case]
    launcher, warp, clang, output = (getattr(args, key).resolve() for key in ("cumetal", "warp", "clang", "output"))
    python = args.python.absolute()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty; use a new directory to preserve earlier evidence")
    commit = subprocess.check_output(["git", "-C", str(warp), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN:
        parser.error(f"expected Warp {PIN}, found {commit}")
    runtime = next((path for path in (
        launcher.parent / "libcumetal.dylib", launcher.parent.parent / "lib/libcumetal.dylib"
    ) if path.is_file()), None)
    if runtime is None:
        parser.error("CuMetal runtime is missing")
    output.mkdir(parents=True, exist_ok=True)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CUMETAL_", "DYLD_", "WARP_"))
           and key not in ("SDKROOT", "PYTHONHOME", "PYTHONPATH", "PYTHONOPTIMIZE")}
    env.update({"CUMETAL_CLANG": str(clang), "CUMETAL_TRACE_GPU": "1", "PYTHONPATH": str(warp),
                "CUMETAL_DEBUG_ERRORS": "1", "CUMETAL_NVRTC_VERBOSE": "1"})
    if args.save_temps:
        env["CUMETAL_NVRTC_SAVE_TEMPS"] = "1"
    artifact_paths = {"launcher_sha256": launcher, "compiler_sha256": launcher.parent / "cumetalc",
                      "runtime_sha256": runtime, "warp_native_sha256": warp / "warp/bin/libwarp.dylib"}
    report = {
        "passed": False, "complete": False, "device": args.device, "warp_commit": commit,
        "manifest_sha256": sha256(args.manifest), "manifest_total": len(manifest["cases"]),
        "selected_total": len(cases), "selected_cases": [case["case"] for case in cases],
        "identity": {key: sha256(path) for key, path in artifact_paths.items()},
        "application_ptx_configuration": False, "save_temps": args.save_temps,
        "cases": [], "counts": {},
        "note": "Fixed original-test manifest; test wall times are not performance measurements. CPU results do not establish GPU compatibility.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for selection in cases:
        name = selection["case"]
        identifier = selection["test"] if args.device == "cuda:0" else selection["test"].removesuffix("cuda_0") + "cpu"
        check, log = output / f"{name}-checks.json", output / f"{name}.log"
        temporary = output / f"temp-{name}"
        temporary.mkdir()
        child_env = dict(env, WARP_CACHE_PATH=str(output / f"warp-cache-{name}"),
                         CUMETAL_CACHE_DIR=str(output / f"cumetal-cache-{name}"), TMPDIR=str(temporary))
        command = [str(launcher), "run", "--ptx", "--", str(python), "-c", CHILD, identifier, args.device, str(check)]
        start, timed_out = time.monotonic(), False
        with log.open("w") as sink:
            process = subprocess.Popen(command, env=child_env, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = process.wait()
        parse_errors, checks, records = [], None, []
        try:
            checks = json.loads(check.read_text())
        except (ValueError, OSError) as error:
            parse_errors.append(f"missing/invalid report: {error}")
        content = log.read_text(errors="replace")
        try:
            records = provenance(content)
        except ValueError as error:
            parse_errors.append(f"invalid GPU trace: {error}")
        errors = parse_errors + validate_result(checks, records, code, timed_out, args.device, identifier)
        status = failure_kind(checks, code, timed_out, errors)
        result = {**selection, "executed_test": identifier, "status": status, "passed": not errors,
                  "returncode": code, "timed_out": timed_out, "errors": errors, "checks": checks,
                  "gpu_launches": len(records), "provenance": records, "log": str(log),
                  "elapsed_seconds": time.monotonic() - start}
        report["cases"].append(result)
        report["counts"] = dict(Counter(item["status"] for item in report["cases"]))
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{len(report['cases'])}/{len(cases)} {status.upper()} {name}: {len(records)} GPU launches", flush=True)
        if errors:
            print("  " + "; ".join(errors), flush=True)
    finalize_report(report, {key: sha256(path) for key, path in artifact_paths.items()})
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "passed": report["passed"], "counts": report["counts"],
                      "run_errors": report["run_errors"], "results": str(output / "results.json")}), flush=True)
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())

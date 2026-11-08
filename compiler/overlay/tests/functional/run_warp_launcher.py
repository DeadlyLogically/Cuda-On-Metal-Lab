#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

PIN = "e6c3ba2d54bb048115760b5cd7a4bb2573329ae7"
CASES = {
    "triangle": "warp.tests.test_triangle_closest_point.TestTriangleClosestPoint.test_triangle_closest_point_cuda_0",
    "scalar_grad": "warp.tests.test_grad.TestGrad.test_scalar_grad_cuda_0",
    "loop_grad": "warp.tests.test_grad.TestGrad.test_for_loop_grad_cuda_0",
    "vec_inplace": "warp.tests.test_vec.TestVec.test_vec_add_inplace_cuda_0",
}
CHILD = r'''
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import warp as wp

identifier, report_path = sys.argv[1:]
assert wp.config.cuda_output is None, "Application must not select PTX"
assert os.environ.get("CUMETAL_DRIVER_JIT") == "runtime-msl"
assert os.environ.get("CUMETAL_PTX_BACKEND") == "cumetal-ir"
assert os.environ.get("CUMETAL_FP64_MODE") == "ieee64"
wp.init()
device = wp.get_device("cuda:0")
assert device.is_cuda and device.name.startswith("Apple "), device
assert device.get_cuda_output_format() == "ptx"
assert wp.get_device("cpu").get_cuda_output_format() is None
assert device.get_cuda_output_format("cubin") == "cubin"
assert device.get_cuda_output_format("ptx") == "ptx"

try:
    wp.config.cuda_output = "cubin"
    assert device.get_cuda_output_format() == "cubin"
    assert device.get_cuda_output_format("ptx") == "ptx"
finally:
    wp.config.cuda_output = None
with patch.dict(os.environ):
    os.environ.pop("CUMETAL_DRIVER_JIT", None)
    assert device.get_cuda_output_format() == "cubin"
    assert device.get_cuda_output_format("ptx") == "ptx"

module_name = identifier.rsplit(".", 2)[0]
module = importlib.import_module(module_name)
checkout = Path(wp.__file__).resolve().parent.parent
source = Path(module.__file__).resolve()
relative = source.relative_to(checkout).as_posix()
original = subprocess.check_output(["git", "-C", str(checkout), "show", "HEAD:" + relative])
assert source.read_bytes() == original, "Original test/calculations were modified"
suite = unittest.defaultTestLoader.loadTestsFromName(identifier)
result = unittest.TextTestRunner(verbosity=2).run(suite)
passed = result.wasSuccessful() and result.testsRun == 1 and not result.skipped and not result.expectedFailures
assert wp.config.cuda_output is None, "Test changed the automatic output selection"
report = {
    "passed": passed,
    "test": identifier,
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
    "expected_failures": len(result.expectedFailures),
    "device": device.name,
    "output_config": wp.config.cuda_output,
    "selected_output": device.get_cuda_output_format(),
    "warp_version": wp.__version__,
    "original_source": relative,
    "source_sha256": hashlib.sha256(original).hexdigest(),
    "source_unchanged": True,
}
Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(0 if passed else 1)
'''

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def provenance(content):
    marker = "CUMETAL_PROVENANCE "
    records = []
    for line in content.splitlines():
        if marker not in line:
            continue
        fields = {}
        for token in shlex.split(line[line.index(marker):]):
            if "=" in token:
                name, value = token.split("=", 1)
                fields[name] = value
        records.append(fields)
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cumetal", required=True, type=Path)
    parser.add_argument("--warp", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--clang", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error("timeout must be between 1 and 600 seconds")
    launcher = args.cumetal.resolve()
    warp = args.warp.resolve()

    python = args.python.absolute()
    clang = args.clang.resolve()
    output = args.output.resolve()
    commit = subprocess.check_output(["git", "-C", str(warp), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN:
        parser.error(f"expected Warp {PIN}, found {commit}")
    output.mkdir(parents=True, exist_ok=True)

    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("CUMETAL_", "DYLD_", "WARP_"))
        and key not in ("SDKROOT", "PYTHONHOME", "PYTHONPATH", "PYTHONOPTIMIZE")
    }
    env.update({
        "CUMETAL_CLANG": str(clang),
        "CUMETAL_TRACE_GPU": "1",
        "PYTHONPATH": str(warp),
    })
    report = {
        "passed": False,
        "warp_commit": commit,
        "launcher_sha256": sha256(launcher),
        "compiler_sha256": sha256(launcher.parent / "cumetalc"),
        "warp_native_sha256": sha256(warp / "warp/bin/libwarp.dylib"),
        "application_ptx_configuration": False,
        "manually_selected_driver_backend_or_precision": False,
        "cases": [],
        "note": "Four original tests of a prepared Warp build; timing is test duration, not a benchmark.",
    }
    runtime = next((path for path in (
        launcher.parent / "libcumetal.dylib", launcher.parent.parent / "lib/libcumetal.dylib"
    ) if path.is_file()), None)
    if runtime is None:
        parser.error("CuMetal runtime is missing beside the launcher or in its installation prefix")
    report["runtime_file"] = str(runtime)
    report["runtime_sha256"] = sha256(runtime)
    for name, identifier in CASES.items():
        check = output / f"{name}-checks.json"
        check.unlink(missing_ok=True)
        log = output / f"{name}.log"
        child_env = dict(env, WARP_CACHE_PATH=str(output / f"warp-cache-{name}"),
                         CUMETAL_CACHE_DIR=str(output / f"cumetal-cache-{name}"))
        command = [str(launcher), "run", "--ptx", "--", str(python), "-c", CHILD, identifier, str(check)]
        start = time.monotonic()
        timed_out = False
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
        errors = []
        checks = None
        try:
            checks = json.loads(check.read_text())
        except (ValueError, OSError) as error:
            errors.append(f"missing/invalid numerical report: {error}")
        try:
            records = provenance(log.read_text(errors="replace"))
        except ValueError as error:
            records = []
            errors.append(f"invalid GPU trace: {error}")
        if code or timed_out:
            errors.append(f"process exit={code}, timeout={timed_out}")
        if not checks or not (
            checks.get("passed") is True and checks.get("tests_run") == 1
            and all(checks.get(key) == 0 for key in ("failures", "errors", "skipped", "expected_failures"))
            and checks.get("source_unchanged") is True
            and checks.get("output_config") is None and checks.get("selected_output") == "ptx"
        ):
            errors.append("unchanged original test did not pass with automatic PTX selection")
        if not records:
            errors.append("no GPU launches observed")
        allowed = {"generic_ptx_lowering", "generic_ptx_lowering_fp64_ieee64"}
        for record in records:
            if not (record.get("event") == "kernel_launch" and record.get("source") == "generic_ptx"
                    and record.get("provenance") in allowed and record.get("device") == "apple_gpu"
                    and record.get("device_name", "").startswith("Apple ")
                    and record.get("launch_success") == "true" and record.get("semantic_quality") == "exact"):
                errors.append(f"unexpected GPU provenance: {record}")
        case = {"case": name, "passed": not errors, "errors": errors, "checks": checks,
                "gpu_launches": len(records), "provenance": records, "log": str(log),
                "elapsed_seconds": time.monotonic() - start}
        report["cases"].append(case)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{'PASS' if not errors else 'FAIL'} {name}: {len(records)} GPU launches", flush=True)
        for error in errors:
            print(error, flush=True)
    report["passed"] = all(case["passed"] for case in report["cases"])
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())

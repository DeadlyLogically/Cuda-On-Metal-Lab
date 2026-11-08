import argparse, os, subprocess, sys, json, hashlib, re
from pathlib import Path
from statistics import median

p = argparse.ArgumentParser()
p.add_argument("--build", type=Path, required=True)
p.add_argument("--lib", type=Path, required=True)
p.add_argument("--baseline-lib", type=Path, required=True)
p.add_argument("--results", type=Path, required=True)
p.add_argument("--t4", type=Path)
p.add_argument("--t4-bench", type=Path)
args = p.parse_args()
root = Path(__file__).resolve().parent
out = args.results.resolve()
out.mkdir(exist_ok=True, parents=True)
lib = args.lib.resolve()
fixtures = root / "fixtures/build"
mac = root / "mac"
results = []
env = {
    **os.environ,
    "PATH": str(args.build.resolve()) + os.pathsep + os.environ["PATH"],
    "CUMETAL_DRIVER_JIT": "runtime-msl",
    "CUMETAL_PTX_BACKEND": "cumetal-ir",
    "CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS": "0",
    "CUMETAL_DISABLE_LLMC_EMULATION": "1",
    "CUMETAL_ENABLE_LLMC_CPU_EMULATION": "0",
    "CUMETAL_SASS_LIFTER": str(root / "lifter/lift.py"),
    "CUMETAL_SASS_NVDISASM": str(mac / "nvdisasm-docker"),
}


def runfile(stem, name, symbol, extra, negative=False):
    args1 = [
        sys.executable,
        str(mac / "run_gpu.py"),
        str(fixtures / (name + ".sm75.cubin")),
        symbol,
        "--out",
        str(out / (stem + ".json")),
        *extra,
    ]
    proc = subprocess.run(
        args1, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    (out / (stem + ".log")).write_text(proc.stdout)
    row = json.loads((out / (stem + ".json")).read_text())
    proof = [
        line
        for line in proc.stdout.splitlines()
        if "CUMETAL_PROVENANCE event=kernel_launch" in line
        and "launch_success=true" in line
        and "device=apple_gpu" in line
    ]
    if negative:
        assert proc.returncode != 0 and not proof and not row["passed"], (
            stem,
            row,
            proof,
        )
        if stem == "atomicSum":
            assert (
                "atomicSum+0xa0: unsupported instruction RED.E.ADD" in proc.stdout
                and "801" in row["error"]
            )
    else:
        assert (
            proc.returncode == 0
            and row["passed"]
            and len(proof) == len(row["rows"]) == 9
            and all(x["oracle_match"] for x in row["rows"])
        ), (stem, proc.stdout[-1200:])
    return row


stock = runfile(
    "stock",
    "vectorAdd",
    "vectorAdd",
    ["--stock", "--library", str(args.baseline_lib.resolve())],
    True,
)
for name in ["vectorAdd", "tripleSum", "elementMultiply", "sameNameMultiply"]:
    mul = name == "sameNameMultiply"
    results.append(
        (
            name,
            runfile(
                name,
                name,
                "vectorAdd" if mul else name,
                ["--library", str(lib)]
                + (["--oracle", "elementMultiply"] if mul else []),
            ),
        )
    )
runfile("atomicSum", "atomicSum", "atomicSum", ["--library", str(lib)], True)
mem = runfile(
    "memory", "vectorAdd", "vectorAdd", ["--from-memory", "--library", str(lib)]
)
assert [x["output_sha256"] for x in results[0][1]["rows"]] == [
    x["output_sha256"] for x in mem["rows"]
]
guard = subprocess.run(
    [sys.executable, str(mac / "guarded_memory.py"), str(lib)],
    check=True,
    capture_output=True,
    env=env,
    text=True,
)
(out / "guarded.json").write_text(guard.stdout)
subprocess.run(
    [
        sys.executable,
        str(root / "bench.py"),
        str(lib),
        str(fixtures / "vectorAdd.sm75.cubin"),
        str(out / "bench.json"),
    ],
    check=True,
    env={**env, "CUMETAL_TRACE_GPU": "0"},
)
summary = {
    "status": "mac-validated",
    "file_cases": sum(len(r["rows"]) for _, r in results),
    "elements": sum(sum(x["n"] for x in r["rows"]) for _, r in results),
    "extra_in_memory_cases": len(mem["rows"]),
    "baseline": stock["error"],
    "unsupported": "atomicSum+0xa0: RED.E.ADD.F32.FTZ.RN.STRONG.GPU; zero launches",
    "sha256_lifter": hashlib.sha256((root / "lifter/lift.py").read_bytes()).hexdigest(),
    "sha256_runtime": hashlib.sha256(lib.read_bytes()).hexdigest(),
}
if args.t4:
    native = json.loads(args.t4.read_text())
    assert len(native["tests"]) == 4 and "Tesla T4" in native["hardware"]
    for name, r in results:
        t = next(x for x in native["tests"] if x["cubin"] == name + ".sm75.cubin")
        assert t["cubin_sha256"] == r["cubin_sha256"]
        for m, v in zip(r["rows"], t["rows"], strict=True):
            assert (m["n"], m["output_sha256"], m["guard_sha256"]) == (
                v["n"],
                v["sha256"],
                v["guard_sha256"],
            )
    summary["status"] = "identical on M2 and T4"
    summary["t4_hardware"] = native["hardware"]
if args.t4_bench:
    m = json.loads((out / "bench.json").read_text())
    t = json.loads(args.t4_bench.read_text())
    assert (m["sha256"], m["cubin_sha256"], m["n"]) == (
        t["sha256"],
        t["cubin_sha256"],
        t["n"],
    )
    summary["warm_latency_us"] = {
        "M2": m["median_us"],
        "T4": t["median_us"],
        "ratio": m["median_us"] / t["median_us"],
        "scope": m["scope"],
        "observations_per_device": len(m["raw_us"]),
    }
(out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(summary)

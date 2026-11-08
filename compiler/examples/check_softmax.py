#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

here = Path(os.environ['PROBE_DIR']).resolve()
mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
assert mode in ("smoke", "sweep")
env = os.environ.copy()
settings = {
    "CUMETAL_TRACE_GPU": "1",
    "CUMETAL_PTX_BACKEND": "cumetal-ir",
    "CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS": "0",
    "CUMETAL_ENABLE_LLMC_CPU_EMULATION": "0",
    "CUMETAL_DISABLE_LLMC_EMULATION": "1",
    "CUMETAL_CACHE_DIR": str(here / "cache"),
}
env.update(settings)
command = [str(here / "softmax_auto"), mode]
log = here / f"{mode}.log"
started = time.monotonic()
timed_out = False
with log.open("w") as sink:
    try:
        result = subprocess.run(command, env=env, stdout=sink, stderr=subprocess.STDOUT, timeout=180)
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = None
elapsed = time.monotonic() - started
content = log.read_text()
records = [line for line in content.splitlines() if line.startswith("CUMETAL_PROVENANCE ")]
summary = re.search(r"AUTO_SOFTMAX_OK launches=(\d+) values=(\d+) max_abs_error=([\deE.+-]+) max_row_sum_error=([\deE.+-]+)", content)
expected_launches = 1 if mode == "smoke" else 240
valid_records = all(
    "device=apple_gpu " in line
    and "launch_success=true " in line
    and "provenance=generic_ptx_lowering " in line
    and "softmax_forward_kernel2" in line
    for line in records
)
passed = returncode == 0 and summary is not None and len(records) == expected_launches and valid_records
if summary:
    passed = passed and int(summary.group(1)) == expected_launches
report = {
    "command": command,
    "environment": settings,
    "returncode": returncode,
    "timed_out": timed_out,
    "wall_seconds": elapsed,
    "log": str(log),
    "binary_sha256": hashlib.sha256((here / "softmax_auto").read_bytes()).hexdigest(),
    "passed": passed,
    "expected_launches": expected_launches,
    "provenance_records": len(records),
    "all_records_generic_apple_gpu_success": valid_records and bool(records),
    "numeric_summary": None if not summary else {
        "launches": int(summary.group(1)),
        "values": int(summary.group(2)),
        "max_abs_error": float(summary.group(3)),
        "max_row_sum_error": float(summary.group(4)),
    },
    "first_provenance_record": records[0] if records else None,
}
(here / f"{mode}.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if not passed:
    print("\n".join(content.splitlines()[-35:]))
    raise SystemExit(1)

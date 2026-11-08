#!/usr/bin/env python3
import hashlib
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

here = Path(os.environ['PROBE_DIR']).resolve()
program = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else 'sweep'
assert program in ('layernorm', 'attention') and mode in ('smoke', 'sweep')
env = os.environ.copy()
settings = dict(CUMETAL_TRACE_GPU='1', CUMETAL_DEBUG_REGISTRATION='1',
    CUMETAL_PTX_BACKEND='cumetal-ir', CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0',
    CUMETAL_ENABLE_LLMC_CPU_EMULATION='0', CUMETAL_DISABLE_LLMC_EMULATION='1',
    CUMETAL_CACHE_DIR=str(here / ('cache-' + program)))
env.update(settings)
if program == 'attention':

    settings['CUMETAL_FP64_MODE'] = 'ieee64'
    env['CUMETAL_FP64_MODE'] = 'ieee64'
command = [str(here / (program + '_auto')), mode]
log = here / (program + '-' + mode + '.log')
start = time.monotonic()
with log.open('w') as sink:
    try:
        r = subprocess.run(command, env=env, stdout=sink, stderr=subprocess.STDOUT, timeout=180)
        code, timeout = r.returncode, False
    except subprocess.TimeoutExpired:
        code, timeout = None, True
content = log.read_text()
lines = [x for x in content.splitlines() if x.startswith('CUMETAL_PROVENANCE ')]
match = re.search('AUTO_' + program.upper() + r'_OK ([^\n]+)', content)
summary = dict(re.findall(r'(\w+)=([^ ]+)', match.group(1))) if match else None
expected_cases = (1 if mode == 'smoke' else 111) if program == 'layernorm' else (1 if mode == 'smoke' else 72)
expected_launches = expected_cases * (1 if program == 'layernorm' else 3)
provenance = [re.search(r' provenance=([^ ]+)', x).group(1) for x in lines]
kernel_counts = Counter(re.search(r' kernel="([^"]+)"', x).group(1) for x in lines)
names = (['layernorm_forward_kernel1' if mode == 'smoke' else 'layernorm_forward_kernel3']
         if program == 'layernorm' else
         ['attention_query_key_kernel1', 'attention_softmax_kernel1', 'attention_value_kernel1'])
valid_kernels = len(kernel_counts) == len(names) and all(
    sum(n for kernel, n in kernel_counts.items() if name in kernel) == expected_cases
    for name in names)
valid = len(lines) == expected_launches and all(
    'device=apple_gpu ' in x and 'launch_success=true ' in x and
    p in ('generic_ptx_lowering', 'generic_ptx_lowering_fp64_emulated', 'generic_ptx_lowering_fp64_wide48', 'generic_ptx_lowering_fp64_ieee64')
    for x, p in zip(lines, provenance))
passed = code == 0 and summary is not None and valid and valid_kernels and int(summary['cases']) == expected_cases and int(summary['launches']) == expected_launches
report = dict(program=program, mode=mode, command=command, environment=settings,
    returncode=code, timed_out=timeout, wall_seconds=time.monotonic()-start,
    log=str(log), passed=passed, expected_cases=expected_cases, expected_launches=expected_launches,
    records=len(lines), kernels=dict(kernel_counts), provenances=sorted(set(provenance)), numeric_summary=summary,
    first_record=lines[0] if lines else None,
    binary_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
    runtime_sha256=hashlib.sha256((Path(os.environ['CUMETAL_BUILD_DIR'])/'libcumetal.dylib').read_bytes()).hexdigest())
(here/(program+'-'+mode+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not passed:
    print('\n'.join(content.splitlines()[-45:])); raise SystemExit(1)

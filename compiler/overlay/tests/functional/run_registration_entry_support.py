#!/usr/bin/env python3

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def run(binary, mode, env):
    result = subprocess.run([str(binary), mode], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=40)
    print(result.stdout, end='')
    if result.returncode == 77:
        raise SystemExit(77)
    require(result.returncode == 0, f'{mode}: test returned {result.returncode}')
    return result.stdout

def main():
    binary = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix='cumetal-entry-support-') as temporary:
        root = Path(temporary)
        tool_log = root / 'offline-tool.log'
        tool = root / 'unavailable-xcrun'
        tool.write_text('''#!/bin/sh
printf '%s\\n' "$@" >> "$CUMETAL_ENTRY_TOOL_LOG"
exit 1
''')
        tool.chmod(0o755)
        env = os.environ.copy()
        env.update(CUMETAL_CACHE_DIR=str(root / 'fp32-cache'),
                   CUMETAL_PTX_BACKEND='cumetal-ir',
                   CUMETAL_FP64_MODE='fast48',
                   CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0',
                   CUMETAL_ENABLE_LLMC_CPU_EMULATION='0',
                   CUMETAL_DISABLE_LLMC_EMULATION='1',
                   CUMETAL_DEBUG_REGISTRATION='1', CUMETAL_TRACE_GPU='1',
                   CUMETAL_TOOL_XCRUN=str(tool),
                   CUMETAL_ENTRY_TOOL_LOG=str(tool_log))
        cases = (
            ('fp32', 'fast48', 'generic_ptx_lowering', 'exact'),
            ('fp64', 'fast48', 'generic_ptx_lowering_fp64_emulated', 'reduced_precision_fp64'),
            ('fp64', 'wide48', 'generic_ptx_lowering_fp64_wide48', 'reduced_precision_fp64'),
            ('fp64', 'ieee64', 'generic_ptx_lowering_fp64_ieee64', 'exact'),
            ('libdevice', 'fast48', 'generic_ptx_lowering', 'exact'),
        )
        for kind, mode, provenance, quality in cases:
            env['CUMETAL_CACHE_DIR'] = str(root / f'{kind}-{mode}-cache')
            env['CUMETAL_FP64_MODE'] = mode
            for phase in ('cold', 'warm'):
                label = f'{kind}/{mode}/{phase}'
                print(f'ENTRY_SUPPORT_PHASE {label}', flush=True)
                log = run(binary, kind, env)
                require(f'ENTRY_SUPPORT_NUMERICAL_OK kind={kind} values=37' in log,
                        f'{label}: missing numerical check')
                traces = [line for line in log.splitlines()
                          if line.startswith('CUMETAL_PROVENANCE ')]
                require(len(traces) == 1, f'{label}: expected one GPU dispatch')
                require(all(item in traces[0] for item in (
                    f'provenance={provenance} ', f'semantic_quality={quality} ',
                    'device=apple_gpu ', 'launch_success=true')),
                    f'{label}: unexpected provenance: {traces[0]}')
                require(not tool_log.exists(), f'{label}: runtime MSL used offline tool')
                cache = Path(env['CUMETAL_CACHE_DIR']) / 'registration-jit'
                sources = list(cache.glob('*.metal'))
                require(len(sources) == 1, f'{label}: expected runtime-MSL cache')
                require(len(list(cache.glob('*.metadata'))) == 1, f'{label}: missing metadata')
                require(not list(cache.glob('*.metallib')), f'{label}: unexpected offline artifact')
                source = sources[0].read_text()
                require(not re.search(r'^\s*#\s*include\s*"', source, re.MULTILINE),
                        f'{label}: runtime source depends on external quoted includes')
                if kind == 'fp64':
                    require('inline ulong cm_fp64_fast_add(' in source and
                            'inline ulong vf64_add_rne(' in source,
                            f'{label}: FP64 support definitions absent')
                else:
                    require('inline ulong cm_fp64_fast_add(' not in source,
                            f'{label}: unrelated FP64 support was included')
                if kind == 'libdevice':
                    require('extern "C" float cm_libdevice_round(float x)' in source,
                            f'{label}: float libdevice helper definition absent')
                if phase == 'cold':
                    require('jit cache miss:' in log, f'{label}: no cache miss observed')
                else:
                    require('jit metadata cache hit:' in log, f'{label}: metadata not restored')
                    require('using direct Metal lowering path' not in log,
                            f'{label}: repeated lowering instead of cache restoration')

        env['CUMETAL_CACHE_DIR'] = str(root / 'unsupported-cache')
        env['CUMETAL_FP64_MODE'] = 'fast48'
        log = run(binary, 'unsupported', env)
        require('ENTRY_SUPPORT_UNSUPPORTED_REFUSED' in log, 'missing explicit refusal')
        require('lower_ptx_to_metal_source failed' in log,
                'unsupported entry did not report a lowering failure')
        require(not tool_log.exists(), 'unsupported PTX incorrectly attempted offline fallback')
        require(not any(line.startswith('CUMETAL_PROVENANCE ') for line in log.splitlines()),
                'unsupported PTX unexpectedly dispatched on GPU')
        require(not list(Path(env['CUMETAL_CACHE_DIR']).rglob('*.metal')),
                'unsupported PTX published a runtime-MSL artifact')
    print('PASS: runtime-MSL FP32/FP64/libdevice cold/warm GPU numerics '
          '(10 launches, 370 values), mode provenance, and unsupported PTX refusal')

if __name__ == '__main__':
    main()

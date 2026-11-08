#!/usr/bin/env python3

import ctypes as c
import mmap
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

PTX = r"""
.version 7.0
.target sm_80
.address_size 64
.extern .func (.param .b32 result) __nv_roundf(.param .b32 value);
.global .align 4 .b8 private_counter[4] = {5, 0, 0, 0};
.visible .global .align 4 .u32 visible_counter = 100;

.visible .entry entry_fp32(
    .param .u64 .ptr .global output,
    .param .u64 .ptr .global input,
    .param .u32 count)
{
    .reg .pred %p;
    .reg .b32 %r<2>;
    .reg .b64 %rd<5>;
    .reg .f32 %f<3>;
    ld.param.u64 %rd0, [output];
    ld.param.u64 %rd1, [input];
    ld.param.u32 %r0, [count];
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p, %r1, %r0;
    @%p bra DONE;
    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    ld.global.f32 %f0, [%rd4];
    add.f32 %f1, %f0, %f0;
    add.f32 %f2, %f1, 0f3F800000;
    st.global.f32 [%rd3], %f2;
DONE:
    ret;
}

.visible .entry entry_byval(
    .param .align 8 .b8 bounds[32],
    .param .f32 scale,
    .param .u64 .ptr .global output)
{
    .reg .pred %p;
    .reg .b32 %r<3>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<6>;
    ld.param.u32 %r0, [bounds+12];
    ld.param.u32 %r1, [bounds+16];
    ld.param.u64 %rd0, [bounds+24];
    ld.param.f32 %f0, [scale];
    ld.param.u64 %rd1, [output];
    mov.u32 %r2, %tid.x;
    cvt.u64.u32 %rd2, %r2;
    setp.ge.u64 %p, %rd2, %rd0;
    @%p bra DONE_BYVAL;
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd1, %rd3;
    cvt.rn.f32.u32 %f1, %r0;
    cvt.rn.f32.u32 %f2, %r1;
    cvt.rn.f32.u32 %f3, %r2;
    mul.f32 %f4, %f0, %f2;
    add.f32 %f5, %f4, %f1;
    add.f32 %f5, %f5, %f3;
    st.global.f32 [%rd4], %f5;
DONE_BYVAL:
    ret;
}

.visible .entry entry_fp64(
    .param .u64 .ptr .global output,
    .param .u64 .ptr .global input,
    .param .u32 count)
{
    .reg .pred %p;
    .reg .b32 %r<2>;
    .reg .b64 %rd<5>;
    .reg .f64 %fd<2>;
    ld.param.u64 %rd0, [output];
    ld.param.u64 %rd1, [input];
    ld.param.u32 %r0, [count];
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p, %r1, %r0;
    @%p bra DONE64;
    mul.wide.u32 %rd2, %r1, 8;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    ld.global.f64 %fd0, [%rd4];
    add.rn.f64 %fd1, %fd0, %fd0;
    st.global.f64 [%rd3], %fd1;
DONE64:
    ret;
}

.visible .entry entry_libdevice(
    .param .u64 .ptr .global output,
    .param .u64 .ptr .global input,
    .param .u32 count)
{
    .reg .pred %p;
    .reg .b32 %r<2>;
    .reg .b64 %rd<5>;
    .reg .f32 %f<2>;
    .param .b32 arg0;
    .param .b32 result0;
    ld.param.u64 %rd0, [output];
    ld.param.u64 %rd1, [input];
    ld.param.u32 %r0, [count];
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p, %r1, %r0;
    @%p bra DONE_LIB;
    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    ld.global.f32 %f0, [%rd4];
    st.param.f32 [arg0], %f0;
    call.uni (result0), __nv_roundf, (arg0);
    ld.param.f32 %f1, [result0];
    st.global.f32 [%rd3], %f1;
DONE_LIB:
    ret;
}

.visible .entry entry_bump7(.param .u64 .ptr .global output)
{
    .reg .b64 %rd;
    .reg .b32 %r<4>;
    ld.param.u64 %rd, [output];
    ld.global.u32 %r0, [private_counter];
    add.u32 %r1, %r0, 7;
    st.global.u32 [private_counter], %r1;
    ld.global.u32 %r2, [visible_counter];
    add.u32 %r3, %r2, %r1;
    st.global.u32 [visible_counter], %r3;
    st.global.u32 [%rd], %r3;
    ret;
}

.visible .entry entry_bump2(.param .u64 .ptr .global output)
{
    .reg .b64 %rd;
    .reg .b32 %r<4>;
    ld.param.u64 %rd, [output];
    ld.global.u32 %r0, [private_counter];
    add.u32 %r1, %r0, 2;
    st.global.u32 [private_counter], %r1;
    ld.global.u32 %r2, [visible_counter];
    add.u32 %r3, %r2, %r1;
    st.global.u32 [visible_counter], %r3;
    st.global.u32 [%rd], %r3;
    ret;
}

.visible .entry entry_unsupported(.param .u64 .ptr .global output)
{
    .reg .b64 %rd;
    .reg .b32 %r<3>;
    ld.param.u64 %rd, [output];
    mov.u32 %r0, %tid.x;
    add.cc.u32 %r1, %r0, 1;
    addc.u32 %r2, %r1, 0;
    st.global.u32 [%rd], %r2;
    ret;
}
"""

P, U32, U64, SIZE = c.c_void_p, c.c_uint32, c.c_uint64, c.c_size_t

DYNAMIC_PTX = r"""
.version 7.0
.target sm_80
.address_size 64
.visible .entry parent(.param .u64 .ptr parent_param_0)
{
    .reg .b64 %rd<8>;
    .reg .b32 %r<3>;
    .reg .pred %p<2>;
    ld.param.b64 %rd0, [parent_param_0];
    {
    .param .b64 param0;
    .param .b64 param1;
    .param .b64 retval0;
    st.param.b64 [param1], 8;
    st.param.b64 [param0], 64;
    call.uni (retval0), cudaGetParameterBuffer, (param0, param1);
    ld.param.b64 %rd1, [retval0];
    }
    setp.eq.b64 %p0, %rd1, 0;
    @%p0 bra DONE_LAUNCH;
    st.b64 [%rd1], %rd0;
    {
    .param .b64 param0;
    .param .b64 param1;
    .param .align 4 .b8 param2[12];
    .param .align 4 .b8 param3[12];
    .param .b32 param4;
    .param .b64 param5;
    .param .b32 retval0;
    st.param.b32 [param2], 1;
    st.param.b32 [param2+4], 1;
    st.param.b32 [param2+8], 1;
    st.param.b32 [param3], 32;
    st.param.b32 [param3+4], 1;
    st.param.b32 [param3+8], 1;
    st.param.b64 [param1], %rd1;
    mov.b64 %rd2, child;
    st.param.b64 [param0], %rd2;
    st.param.b64 [param5], 0;
    st.param.b32 [param4], 0;
    call.uni (retval0), cudaLaunchDevice, (param0, param1, param2, param3, param4, param5);
    ld.param.b32 %r0, [retval0];
    }
DONE_LAUNCH:
    ret;
}
.visible .entry child(.param .u64 .ptr child_param_0)
{
    ret;
}
"""

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def child(library, kind):
    lib = c.CDLL(str(library))

    def api(name, types, *args, expected=0):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, c.c_int
        status = fn(*args)
        if status == 100 and name in ('cuInit', 'cuDeviceGetCount'):
            raise SystemExit(77)
        require(status == expected, f'{name}: {status}, expected {expected}')

    api('cuInit', [U32], 0)
    count = c.c_int()
    api('cuDeviceGetCount', [c.POINTER(c.c_int)], c.byref(count))
    if not count.value:
        return 77
    ctx, module = P(), P()
    api('cuCtxCreate', [c.POINTER(P), U32, c.c_int], c.byref(ctx), 0, 0)
    if kind == 'image_limit':

        limit = 64 * 1024 * 1024
        pages = mmap.mmap(-1, limit + mmap.PAGESIZE)
        base = c.addressof(c.c_char.from_buffer(pages))
        protect = c.CDLL(None).mprotect
        protect.argtypes, protect.restype = [P, SIZE, c.c_int], c.c_int
        c.memset(base, ord(' '), limit)
        c.memmove(base, PTX.encode(), len(PTX.encode()))
        require(protect(base + limit, mmap.PAGESIZE, 0) == 0, 'image guard setup failed')
        try:
            api('cuModuleLoadData', [c.POINTER(P), P], c.byref(module), base, expected=200)
            require(not module.value, 'over-limit image returned a module')
        finally:
            require(protect(base + limit, mmap.PAGESIZE, mmap.PROT_READ | mmap.PROT_WRITE) == 0,
                    'image guard cleanup failed')
            pages.close()
            api('cuCtxDestroy', [P], ctx)
        print('DRIVER_RUNTIME_MSL_IMAGE_LIMIT_REFUSED')
        return 0
    test_dir = Path(__file__).resolve().parent
    sources = {
        'printf': ('printf_scale', test_dir / 'reference/printf_scale.ptx'),
        'heap': ('heapk', test_dir / 'fixtures/ptx_device_heap.ptx'),
    }
    source = sources[kind][1].read_text() if kind in sources else DYNAMIC_PTX if kind == 'dynamic' else PTX
    if kind == 'large_ptx':

        source += '\n//' + ' ' * (2 * 1024 * 1024) + '\n'
    image = c.create_string_buffer(source.encode())
    api('cuModuleLoadData', [c.POINTER(P), P], c.byref(module), image)
    buffers = []
    guarded_pages = None
    protected_address = 0
    protect = c.CDLL(None).mprotect
    protect.argtypes, protect.restype = [P, SIZE, c.c_int], c.c_int

    def function(name, expected=0):
        handle = P()
        api('cuModuleGetFunction', [c.POINTER(P), P, c.c_char_p],
            c.byref(handle), module, name.encode(), expected=expected)
        return handle

    def allocate(values):
        ptr = U64()
        api('cuMemAlloc', [c.POINTER(U64), SIZE], c.byref(ptr), c.sizeof(values))
        buffers.append(ptr)
        api('cuMemcpyHtoD', [U64, P, SIZE], ptr, values, c.sizeof(values))
        return ptr

    def download(values, ptr):
        api('cuMemcpyDtoH', [P, U64, SIZE], values, ptr, c.sizeof(values))

    def launch(handle, args, threads=64):

        params = (P * len(args))(*(c.cast(c.byref(v), P) for v in args))
        api('cuLaunchKernel', [P, U32, U32, U32, U32, U32, U32, U32, P, c.POINTER(P), P],
            handle, 1, 1, 1, threads, 1, 1, 0, None, params, None)
        api('cuCtxSynchronize', [])

    try:
        function('missing_entry', expected=500)
        if kind == 'unsupported':
            function('entry_unsupported', expected=801)
            print('DRIVER_RUNTIME_MSL_UNSUPPORTED_REFUSED')
        elif kind in ('printf', 'heap', 'dynamic'):
            function('parent' if kind == 'dynamic' else sources[kind][0], expected=801)
            print(f'DRIVER_RUNTIME_MSL_RESOURCE_REFUSED kind={kind}')
        elif kind == 'packed':
            values = (c.c_float * 2)(1.0, 2.0)
            output, input_ = allocate(values), allocate(values)
            packed = (U64 * 3)(output.value, input_.value, 2)
            size = SIZE(c.sizeof(packed))
            extra = (P * 5)(P(1), c.cast(packed, P), P(2), c.cast(c.byref(size), P), None)
            api('cuLaunchKernel', [P, U32, U32, U32, U32, U32, U32, U32, P, P, P],
                function('entry_fp32'), 1, 1, 1, 32, 1, 1, 0, None, None, extra, expected=801)
            print('DRIVER_RUNTIME_MSL_PACKED_REFUSED')
        elif kind == 'globals':
            address, size = U64(), SIZE()

            api('cuModuleGetGlobal', [c.POINTER(U64), c.POINTER(SIZE), P, c.c_char_p],
                c.byref(address), c.byref(size), module, b'visible_counter')
            require(address.value != 0 and size.value == 4, 'invalid global address/size')
            scalar = (U32 * 1)()
            download(scalar, address)
            require(scalar[0] == 100, f'initial global value: {scalar[0]}')
            scalar[0] = 900
            api('cuMemcpyHtoD', [U64, P, SIZE], address, scalar, 4)
            guard = 0xA5A5A5A5
            result = (U32 * 3)(guard, 0, guard)
            output = allocate(result)
            output_arg = U64(output.value + 4)
            observed = []
            for entry, expected in [('entry_bump7', 912), ('entry_bump2', 926), ('entry_bump7', 947)]:
                launch(function(entry), [output_arg], threads=1)
                download(result, output)
                require(list(result) == [guard, expected, guard], f'{entry}: {list(result)}')
                download(scalar, address)
                require(scalar[0] == expected, f'{entry}: global/launch storage differs')
                observed.append(result[1])
            api('cuModuleGetGlobal', [c.POINTER(U64), c.POINTER(SIZE), P, c.c_char_p],
                c.byref(address), c.byref(size), module, b'missing_global', expected=500)
            print(f'DRIVER_RUNTIME_MSL_GLOBALS_OK {observed}')
        else:

            if kind == 'fp64':
                os.environ['CUMETAL_FP64_MODE'] = 'fast48'
            dtype = c.c_double if kind == 'fp64' else c.c_float
            n, guard, sentinel = 37, 8, -777.25
            values = [(i - 18) * 0.25 for i in range(n)]
            if kind == 'fp64':
                values[0:3] = [2.0 ** 500, 2.0 ** -500, 1.0 + 2.0 ** -52]
            host_in = (dtype * (n + 2 * guard))(*([sentinel] * guard + values + [sentinel] * guard))
            original_input = bytes(host_in)
            host_out = (dtype * len(host_in))(*([sentinel] * len(host_in)))
            dev_in, dev_out = allocate(host_in), allocate(host_out)
            input_arg = U64(dev_in.value + guard * c.sizeof(dtype))
            output_arg = U64(dev_out.value + guard * c.sizeof(dtype))

            guarded_pages = mmap.mmap(-1, 2 * mmap.PAGESIZE)
            base = c.addressof(c.c_char.from_buffer(guarded_pages))
            protected_address = base + mmap.PAGESIZE
            count_arg = U32.from_address(protected_address - c.sizeof(U32))
            count_arg.value = n
            require(protect(protected_address, mmap.PAGESIZE, 0) == 0, 'guard page setup failed')
            if kind == 'byval':
                class Bounds(c.Structure):
                    _fields_ = [('shape', c.c_int32 * 4), ('ndim', c.c_int32), ('size', U64)]
                require(c.sizeof(Bounds) == 32 and Bounds.size.offset == 24, 'host by-value layout')
                bounds = Bounds((c.c_int32 * 4)(n, 1, 1, 8), 3, n)
                args = [bounds, c.c_float(10.0), output_arg]
                expected_values = [38.0 + i for i in range(n)]
            else:
                args = [output_arg, input_arg, count_arg]
                if kind == 'libdevice':
                    import math
                    expected_values = [math.copysign(math.floor(abs(v) + 0.5), v) for v in values]
                elif kind == 'fp64':
                    expected_values = [v + v for v in values]
                else:
                    expected_values = [v * 2 + 1 for v in values]
            expected = [sentinel] * guard + expected_values + [sentinel] * guard
            for phase in ('cold', 'warm'):

                for i in range(len(host_out)):
                    host_out[i] = sentinel
                api('cuMemcpyHtoD', [U64, P, SIZE], dev_out, host_out, c.sizeof(host_out))
                launch(function('entry_fp32' if kind == 'large_ptx' else 'entry_' + kind), args)
                download(host_out, dev_out)
                download(host_in, dev_in)
                require(list(host_out) == expected, f'{kind}/{phase}: numeric output or guards differ')
                require(bytes(host_in) == original_input, f'{kind}/{phase}: input modified')
                print(f'DRIVER_RUNTIME_MSL_NUMERICAL_OK kind={kind} phase={phase} values={n}')
    finally:
        if guarded_pages is not None:
            require(protect(protected_address, mmap.PAGESIZE, mmap.PROT_READ | mmap.PROT_WRITE) == 0,
                    'guard page cleanup failed')
            guarded_pages.close()
        for ptr in buffers:
            api('cuMemFree', [U64], ptr)
        api('cuModuleUnload', [P], module)
        api('cuCtxDestroy', [P], ctx)
    return 0

def main():
    library = Path(sys.argv[1]).resolve()
    if len(sys.argv) == 4 and sys.argv[2] == '--child':
        return child(library, sys.argv[3])
    with tempfile.TemporaryDirectory(prefix='cumetal-driver-msl-') as temporary:
        root = Path(temporary)
        tool_log = root / 'offline-tools.log'
        tool = root / 'forbidden-tool'
        tool.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$CUMETAL_DRIVER_TOOL_LOG"\nexit 1\n')
        tool.chmod(0o755)
        env = os.environ.copy()
        env.update(CUMETAL_DRIVER_JIT='runtime-msl', CUMETAL_PTX_BACKEND='cumetal-ir',
                   CUMETAL_FP64_MODE='ieee64', CUMETAL_TRACE_GPU='1',
                   CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0',
                   CUMETAL_ENABLE_LLMC_CPU_EMULATION='0', CUMETAL_DISABLE_LLMC_EMULATION='1',
                   CUMETAL_TOOL_XCRUN=str(tool), CUMETAL_DRIVER_TOOL_LOG=str(tool_log),
                   CUMETAL_NVRTC_COMPILER=str(tool), CUMETAL_CACHE_DIR=str(root / 'cache'),
                   CUMETAL_DEBUG_ERRORS='1')
        negative_cases = ('unsupported', 'printf', 'heap', 'dynamic', 'packed', 'image_limit')
        for kind in ('fp32', 'large_ptx', 'byval', 'fp64', 'libdevice', 'globals', *negative_cases):
            print(f'DRIVER_RUNTIME_MSL_CASE {kind}', flush=True)
            result = subprocess.run([sys.executable, __file__, str(library), '--child', kind],
                                    env=env, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=45)
            print(result.stdout, end='')
            if result.returncode == 77:
                return 77
            require(result.returncode == 0, f'{kind}: subprocess returned {result.returncode}')
            require(not tool_log.exists(), f'{kind}: attempted offline compiler')
            traces = [s for s in result.stdout.splitlines() if s.startswith('CUMETAL_PROVENANCE ')]
            expected_launches = 0 if kind in negative_cases else 3 if kind == 'globals' else 2
            require(len(traces) == expected_launches, f'{kind}: unexpected launch count {len(traces)}')
            provenance = 'generic_ptx_lowering_fp64_ieee64' if kind == 'fp64' else 'generic_ptx_lowering'
            for line in traces:
                require(all(s in line for s in (f'provenance={provenance} ',
                        'device=apple_gpu ', 'launch_success=true', 'semantic_quality=exact ')),
                        f'{kind}: unexpected GPU provenance: {line}')
            if kind in ('fp32', 'large_ptx', 'byval', 'fp64', 'libdevice'):
                require('compile_cache_hit=false' in traces[0], f'{kind}: missing cold compilation')
                require('compile_cache_hit=true' in traces[1], f'{kind}: missing warm pipeline reuse')
                require(len(re.findall(r'DRIVER_RUNTIME_MSL_NUMERICAL_OK', result.stdout)) == 2,
                        f'{kind}: incomplete numeric checks')
            elif kind == 'globals':
                require('DRIVER_RUNTIME_MSL_GLOBALS_OK [912, 926, 947]' in result.stdout,
                        'module globals lost state across selected entries')
            elif kind == 'unsupported':
                require('DRIVER_RUNTIME_MSL_UNSUPPORTED_REFUSED' in result.stdout,
                        'selected unsupported PTX was not refused')
            elif kind == 'packed':
                require('DRIVER_RUNTIME_MSL_PACKED_REFUSED' in result.stdout,
                        'packed launch should be explicitly unsupported')
            elif kind == 'image_limit':
                require('DRIVER_RUNTIME_MSL_IMAGE_LIMIT_REFUSED' in result.stdout,
                        'over-limit PTX input was not refused')
            else:
                require(f'DRIVER_RUNTIME_MSL_RESOURCE_REFUSED kind={kind}' in result.stdout,
                        f'{kind}: missing resource refusal')

                require('require registration-owned resources' in result.stdout,
                        f'{kind}: did not reach the post-lowering resource check')
    print('PASS: lazy Driver runtime MSL, typed/by-value ABI, FP64/libdevice, '
          'shared globals, bounded large PTX, cold/warm GPU numerics and selected-entry refusal (13 launches)')
    return 0

if __name__ == '__main__':
    sys.exit(main())

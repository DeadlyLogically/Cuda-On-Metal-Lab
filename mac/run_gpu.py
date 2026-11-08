#!/usr/bin/env python3

import argparse
import ctypes as C
import hashlib
import json
import os
import struct
import sys
import time
from contextlib import ExitStack
from pathlib import Path

here = Path(__file__).resolve().parent
U64, U32, I32, PTR = C.c_uint64, C.c_uint32, C.c_int32, C.c_void_p
SIZES = (1, 31, 32, 33, 255, 256, 257, 1003, 50000)
CANARY = -12345.25


class Driver:
    def __init__(self, library):
        self.library = C.CDLL(str(library))
        signatures = {
            "cuInit": [U32],
            "cuDeviceGetName": [PTR, I32, I32],
            "cuCtxCreate": [C.POINTER(PTR), U32, I32],
            "cuCtxDestroy": [PTR],
            "cuCtxSynchronize": [],
            "cuModuleLoad": [C.POINTER(PTR), C.c_char_p],
            "cuModuleLoadData": [C.POINTER(PTR), PTR],
            "cuModuleGetFunction": [C.POINTER(PTR), PTR, C.c_char_p],
            "cuModuleUnload": [PTR],
            "cuMemAlloc": [C.POINTER(U64), C.c_size_t],
            "cuMemFree": [U64],
            "cuMemcpyHtoD": [U64, PTR, C.c_size_t],
            "cuMemcpyDtoH": [PTR, U64, C.c_size_t],
            "cuLaunchKernel": [
                PTR,
                U32,
                U32,
                U32,
                U32,
                U32,
                U32,
                U32,
                PTR,
                C.POINTER(PTR),
                PTR,
            ],
        }
        for name, types in signatures.items():
            fn = getattr(self.library, name)
            fn.argtypes = types
            fn.restype = I32

    def call(self, name, *args):
        status = getattr(self.library, name)(*args)
        if status:
            raise RuntimeError(f"{name} returned {status}")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(name, a, b):
    if name == "vectorAdd":
        return a + b
    if name == "elementMultiply":
        return a * b
    if name == "tripleSum":
        return (a + b) + b
    raise ValueError(f"unknown test oracle {name}")


def run_size(driver, kernel, operation, count):
    block = 256
    grid = (count + block - 1) // block
    capacity = grid * block + 16
    left = [((i * 17 + 3) % 251 - 125) / 8 for i in range(capacity)]
    right = [((i * 29 + 7) % 239 - 119) / 16 for i in range(capacity)]
    storage = C.c_float * capacity
    with ExitStack() as resources:
        addresses = []
        for values in (left, right, [CANARY] * capacity):
            raw = storage(*values)
            device = U64()
            driver.call("cuMemAlloc", C.byref(device), C.sizeof(raw))
            resources.callback(driver.call, "cuMemFree", device)
            driver.call("cuMemcpyHtoD", device, raw, C.sizeof(raw))
            addresses.append(device)
        arg_values = [*addresses, I32(count)]
        args = (PTR * len(arg_values))(*(C.addressof(x) for x in arg_values))
        start = time.perf_counter_ns()
        driver.call(
            "cuLaunchKernel", kernel, grid, 1, 1, block, 1, 1, 0, None, args, None
        )
        driver.call("cuCtxSynchronize")
        sync_ns = time.perf_counter_ns() - start
        buffer = storage()
        driver.call("cuMemcpyDtoH", buffer, addresses[2], C.sizeof(buffer))
        actual = bytes(buffer)
        reference = b"".join(
            struct.pack("<f", ref(operation, left[i], right[i])) for i in range(count)
        )
        payload = actual[: 4 * count]
        tail = actual[4 * count :]
        if tail != struct.pack("<f", CANARY) * (capacity - count):
            raise ValueError(f"{operation}/{count}: tail modified")
        if payload != reference:
            idx = next(
                i
                for i in range(count)
                if reference[4 * i : 4 * i + 4] != payload[4 * i : 4 * i + 4]
            )
            raise ValueError(
                f"{operation}/{count}: first mismatch at {idx}, got {buffer[idx]}, expected {ref(operation, left[idx], right[idx])}"
            )
        return {
            "n": count,
            "output_sha256": sha(payload),
            "guard_sha256": sha(tail[:64]),
            "guard_count_checked": capacity - count,
            "oracle_match": True,
            "launch_sync_ns": sync_ns,
        }


def run(args):
    if sys.byteorder != "little":
        raise ValueError("FP32 byte comparison requires a little endian machine")
    library = args.library.resolve(strict=True)
    cubin = args.cubin.resolve(strict=True)
    driver = Driver(library)
    driver.call("cuInit", 0)
    device = C.create_string_buffer(256)
    driver.call("cuDeviceGetName", device, len(device), 0)
    with ExitStack() as resources:
        ctx = PTR()
        driver.call("cuCtxCreate", C.byref(ctx), 0, 0)
        resources.callback(driver.call, "cuCtxDestroy", ctx)
        module = PTR()
        loaded_at = time.perf_counter_ns()
        if args.from_memory:
            raw = C.create_string_buffer(cubin.read_bytes())
            driver.call("cuModuleLoadData", C.byref(module), raw)
        else:
            driver.call("cuModuleLoad", C.byref(module), os.fsencode(cubin))
        resources.callback(driver.call, "cuModuleUnload", module)
        fn = PTR()
        driver.call("cuModuleGetFunction", C.byref(fn), module, args.entry.encode())
        load_ns = time.perf_counter_ns() - loaded_at
        rows = [run_size(driver, fn, args.oracle or args.entry, n) for n in args.sizes]
    return {
        "device": device.value.decode(),
        "entry": args.entry,
        "library_sha256": sha(library.read_bytes()),
        "load_api": ("cuModuleLoadData" if args.from_memory else "cuModuleLoad"),
        "cubin_sha256": sha(cubin.read_bytes()),
        "load_get_function_ns": load_ns,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cubin", type=Path)
    parser.add_argument("entry")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--oracle", choices=["vectorAdd", "elementMultiply", "tripleSum"]
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--from-memory", action="store_true")
    parser.add_argument("--stock", action="store_true")
    args = parser.parse_args()
    os.environ.update(
        CUMETAL_DRIVER_JIT="runtime-msl",
        CUMETAL_TRACE_GPU="1",
        CUMETAL_PTX_BACKEND="cumetal-ir",
        CUMETAL_DEBUG_ERRORS="1",
        CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS="0",
        CUMETAL_ENABLE_LLMC_CPU_EMULATION="0",
        CUMETAL_DISABLE_LLMC_EMULATION="1",
    )
    if args.stock:
        os.environ.pop("CUMETAL_SASS_LIFTER", None)
    else:
        os.environ.setdefault(
            "CUMETAL_SASS_LIFTER", str(here.parent / "lifter/lift.py")
        )
        os.environ.setdefault("CUMETAL_SASS_NVDISASM", str(here / "nvdisasm-docker"))
        os.environ["CUMETAL_SASS_PYTHON"] = sys.executable
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        args.out.write_text(
            json.dumps({"passed": False, "error": str(exc)}, indent=2) + "\n"
        )
        raise
    report["passed"] = True
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.entry} {report['device']} {len(report['rows'])} sizes passed")


if __name__ == "__main__":
    main()

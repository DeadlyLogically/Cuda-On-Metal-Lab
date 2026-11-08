import ctypes as c
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np

SIZES = [1, 31, 32, 33, 255, 256, 257, 1003, 50000]
P, U, I, D = c.c_void_p, c.c_uint, c.c_int, c.c_uint64
DLL = c.CDLL("libcuda.so.1")


def call(name, argtypes, *args):
    fn = getattr(DLL, name)
    fn.argtypes, fn.restype = argtypes, I
    err = fn(*args)
    if err:
        raise RuntimeError(f"{name}: CUDA error {err}")


def sha(buf):
    return hashlib.sha256(buf).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def inputs(n):
    i = np.arange(n, dtype=np.int32)
    return (((i * 17 + 3) % 251 - 125) / 8).astype("<f4"), (
        ((i * 29 + 7) % 239 - 119) / 16
    ).astype("<f4")


def run(path, symbol, reference):
    blob = path.read_bytes()
    mod = P()
    fn = P()
    call("cuModuleLoad", [c.POINTER(P), c.c_char_p], c.byref(mod), bytes(path))
    try:
        call(
            "cuModuleGetFunction",
            [c.POINTER(P), P, c.c_char_p],
            c.byref(fn),
            mod,
            symbol.encode(),
        )
        rows = []
        for n in SIZES:
            a, b = inputs(n)
            out = np.full(n + 16, -12345.25, dtype="<f4")
            expected = reference(a, b)
            alloc = []
            try:
                for data in (a, b, out):
                    ptr = D()
                    call(
                        "cuMemAlloc_v2",
                        [c.POINTER(D), c.c_size_t],
                        c.byref(ptr),
                        data.nbytes,
                    )
                    alloc.append(ptr)
                    call(
                        "cuMemcpyHtoD_v2",
                        [D, P, c.c_size_t],
                        ptr,
                        data.ctypes.data,
                        data.nbytes,
                    )
                count = I(n)
                argv = (P * 4)(*(c.addressof(x) for x in (*alloc, count)))

                def launch():
                    call(
                        "cuLaunchKernel",
                        [P, U, U, U, U, U, U, U, P, c.POINTER(P), c.POINTER(P)],
                        fn,
                        (n + 255) // 256,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        None,
                        argv,
                        None,
                    )

                launch()
                call("cuCtxSynchronize", [])
                start = perf_counter()
                launch()
                call("cuCtxSynchronize", [])
                us = (perf_counter() - start) * 1e6
                call(
                    "cuMemcpyDtoH_v2",
                    [P, D, c.c_size_t],
                    out.ctypes.data,
                    alloc[2],
                    out.nbytes,
                )
                assert (
                    out[:n].tobytes() == np.asarray(expected, dtype="<f4").tobytes()
                ), (symbol, n, "values")
                assert (out[n:] == -12345.25).all(), (symbol, n, "guards")
                rows.append(
                    dict(
                        n=n,
                        sha256=sha(out[:n].tobytes()),
                        guard_sha256=sha(out[n:].tobytes()),
                        warm_host_launch_and_sync_us=round(us, 4),
                    )
                )
            finally:
                for ptr in alloc:
                    call("cuMemFree_v2", [D], ptr)
    finally:
        call("cuModuleUnload", [P], mod)
    return dict(symbol=symbol, cubin=path.name, cubin_sha256=sha(blob), rows=rows)


def reference(symbol):
    return {
        "vectorAdd": lambda a, b: a + b,
        "elementMultiply": lambda a, b: a * b,
        "tripleSum": lambda a, b: (a + b) + b,
    }[symbol]


def main(folder):
    query = command(
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader",
    )
    assert "T4" in query and "7.5" in query, query
    dev = I()
    ctx = P()
    call("cuInit", [U], 0)
    call("cuDeviceGet", [c.POINTER(I), I], c.byref(dev), 0)
    call("cuDevicePrimaryCtxRetain", [c.POINTER(P), I], c.byref(ctx), dev)
    call("cuCtxSetCurrent", [P], ctx)
    meta = {
        "hardware": query,
        "platform": platform.platform(),
        "block": 256,
        "guards": 16,
        "tests": [],
    }
    try:
        for item in json.loads((folder / "oracle-input.json").read_text()):
            sym = item["symbol"]
            p = folder / item["file"]
            assert sha(p.read_bytes()) == item["sha256"]
            meta["tests"].append(run(p, sym, reference(item.get("reference", sym))))
    finally:
        call("cuCtxSetCurrent", [P], None)
        call("cuDevicePrimaryCtxRelease_v2", [I], dev)
    out = folder / "sass-sm75-t4.json"
    out.write_text(json.dumps(meta, indent=2))
    print(
        "SASS_T4_OK",
        query,
        [(t["symbol"], len(t["rows"])) for t in meta["tests"]],
        sha(out.read_bytes()),
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

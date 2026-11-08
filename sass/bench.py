import ctypes as c
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns


def run(library, path):
    cuda = c.CDLL(library)
    P = c.c_void_p
    U = c.c_uint
    I = c.c_int
    D = c.c_uint64
    size = c.c_size_t

    def bind(symbol, args):
        name = (
            (symbol + "_v2")
            if library == "libcuda.so.1"
            and symbol
            in {
                "cuCtxCreate",
                "cuCtxDestroy",
                "cuMemAlloc",
                "cuMemFree",
                "cuMemcpyHtoD",
                "cuMemcpyDtoH",
            }
            else symbol
        )
        f = getattr(cuda, name)
        f.argtypes = args
        f.restype = I

        def call(*a):
            e = f(*a)
            if e:
                raise RuntimeError(f"{name}: {e}")

        return call

    init = bind("cuInit", [U])
    ctx_create = bind("cuCtxCreate", [c.POINTER(P), U, I])
    ctx_destroy = bind("cuCtxDestroy", [P])
    device_name = bind("cuDeviceGetName", [P, I, I])
    load = bind("cuModuleLoad", [c.POINTER(P), c.c_char_p])
    get = bind("cuModuleGetFunction", [c.POINTER(P), P, c.c_char_p])
    unload = bind("cuModuleUnload", [P])
    alloc = bind("cuMemAlloc", [c.POINTER(D), size])
    free = bind("cuMemFree", [D])
    h2d = bind("cuMemcpyHtoD", [D, P, size])
    d2h = bind("cuMemcpyDtoH", [P, D, size])
    launch = bind("cuLaunchKernel", [P, U, U, U, U, U, U, U, P, c.POINTER(P), P])
    sync = bind("cuCtxSynchronize", [])
    ctx = P()
    mod = P()
    fn = P()
    dev = c.create_string_buffer(256)
    init(0)
    ctx_create(c.byref(ctx), 0, 0)
    device_name(dev, len(dev), 0)
    gpu = dev.value.decode()
    load(c.byref(mod), bytes(path))
    get(c.byref(fn), mod, b"vectorAdd")
    n = 50000
    block = 256
    grid = (n + block - 1) // block
    cap = grid * block + 16
    arr = c.c_float * cap
    a = arr(*[((i * 17 + 3) % 251 - 125) / 8 for i in range(cap)])
    b = arr(*[((i * 29 + 7) % 239 - 119) / 16 for i in range(cap)])
    out = arr(*([-12345.25] * cap))
    ptrs = []
    try:
        for host in (a, b, out):
            d = D()
            alloc(c.byref(d), c.sizeof(host))
            ptrs.append(d)
            h2d(d, host, c.sizeof(host))
        count = I(n)
        args = (P * 4)(*(c.addressof(x) for x in (*ptrs, count)))

        def once():
            t = perf_counter_ns()
            launch(fn, grid, 1, 1, block, 1, 1, 0, None, args, None)
            sync()
            return (perf_counter_ns() - t) / 1000

        for _ in range(5):
            once()
        values = [once() for _ in range(51)]
        d2h(out, ptrs[-1], c.sizeof(out))
        expected = arr(*([a[i] + b[i] for i in range(n)] + [-12345.25] * (cap - n)))
        assert bytes(out) == bytes(expected)
        return {
            "hardware": gpu,
            "cubin_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n": n,
            "symbol": "vectorAdd",
            "warmups": 5,
            "scope": "host wall microseconds from one cuLaunchKernel through cuCtxSynchronize; inputs on GPU, no file load, transfer or trace",
            "median_us": median(values),
            "p10_us": sorted(values)[5],
            "p90_us": sorted(values)[45],
            "raw_us": values,
            "sha256": hashlib.sha256(bytes(out)[: 4 * n]).hexdigest(),
        }
    finally:
        for d in ptrs:
            free(d)
        unload(mod)
        ctx_destroy(ctx)


if __name__ == "__main__":
    res = run(sys.argv[1], Path(sys.argv[2]).resolve())
    Path(sys.argv[3]).write_text(json.dumps(res, indent=2))
    print(
        "SASS_BENCH",
        res["hardware"],
        res["median_us"],
        res["p10_us"],
        res["p90_us"],
        res["sha256"],
    )

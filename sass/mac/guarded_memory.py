import ctypes as c, mmap, os, sys, json
from pathlib import Path
from run_gpu import Driver, PTR

m = Path(__file__).resolve().parent
r = m.parent
os.environ.update(
    CUMETAL_DRIVER_JIT="runtime-msl",
    CUMETAL_SASS_LIFTER=str(r / "lifter/lift.py"),
    CUMETAL_SASS_NVDISASM=str(m / "nvdisasm-docker"),
    CUMETAL_SASS_PYTHON=sys.executable,
)
libc = c.CDLL(None)
libc.mprotect.argtypes = [PTR, c.c_size_t, c.c_int]


def test(library):
    gpu = Driver(library)
    ctx = PTR()
    gpu.call("cuInit", 0)
    gpu.call("cuCtxCreate", c.byref(ctx), 0, 0)
    data = (r / "fixtures/build/vectorAdd.sm75.cubin").read_bytes()
    page = mmap.PAGESIZE
    result = {}
    try:
        for n in (4, 63, 64, len(data) - 1, len(data)):
            with mmap.mmap(-1, page * 2) as mem:
                addr = c.addressof(c.c_char.from_buffer(mem))
                start = addr + page - n
                c.memmove(start, data[:n], n)
                assert libc.mprotect(addr + page, page, 0) == 0
                try:
                    mod = PTR()
                    code = gpu.library.cuModuleLoadData(c.byref(mod), start)
                    expected = 0 if n == len(data) else 200
                    assert code == expected and bool(mod) == (expected == 0), (n, code)
                    if mod:
                        gpu.call("cuModuleUnload", mod)
                    result[n] = code
                finally:
                    assert libc.mprotect(addr + page, page, 3) == 0
    finally:
        gpu.call("cuCtxDestroy", ctx)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test(Path(sys.argv[1]).resolve())

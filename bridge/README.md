# Linux CUDA runtime bridge

The guest process loads an original NVIDIA Linux ARM64 runtime library inside Docker. Its CUDA calls reach a small guest driver over the CUDA Driver API. The driver sends requests to a macOS service, which calls CuMetal to allocate GPU buffers, translate PTX and launch work on Apple Silicon.

This prototype implements one device, one context, explicit memory transfers and one-dimensional kernel launches with pointer arguments. The host owns the GPU memory. Device attributes returned to Linux describe the prototype, including its supported launch geometry. The private interfaces are pinned to the tested release of NVIDIA’s runtime.

## What the probe checks

The baseline recorded CUDA error 35 when the original runtime tried to allocate memory without the bridge. The runtime probe now resolves `cudaMalloc` in NVIDIA’s original library, allocates and copies the arrays, and verifies that it can read back the same values. It then uses the guest Driver API to load a PTX kernel, launch the addition on the Apple M2 and synchronize. The result returned to Linux is `[5, 7, 9]`. CuMetal independently records a GPU dispatch. [Baseline](evidence/baseline.json) · [Runtime checks](evidence/runtime/result.json) · [Guest trace](evidence/runtime/guest.log) · [Host trace](evidence/runtime/host.log)

The driver-only probe runs the same addition through the guest Driver API. Both evidence directories include the exact request/reply transcript and the hashes of the source files used for their run. The host log omits the startup line containing the local sockets and library paths. [Driver checks](evidence/driver/result.json)

## Run on Apple Silicon

Use a Mac with an accessible Metal GPU, Python 3.10 or newer, Docker Desktop running a Linux ARM64 engine, and a CuMetal build containing the runtime Metal shader compiler path. The included CuMetal build recipe pins the underlying revision and configuration. [CuMetal build instructions](../sass/README.md)

From this directory, set `CUMETAL_LIB` to the absolute path of the built `libcumetal.dylib`. The fetch command verifies both NVIDIA’s release archive and the runtime file before saving it locally. Use fresh output directories for each run.

```sh
export CUMETAL_LIB="/absolute/path/to/libcumetal.dylib"
python3 orchestration/fetch_runtime.py
docker build --platform linux/arm64 -t cumetal-bridge-lab:bookworm image
python3 orchestration/run_case.py driver runs/driver --cumetal "$CUMETAL_LIB"
python3 orchestration/run_case.py runtime runs/runtime --private --cumetal "$CUMETAL_LIB"
```

Each command writes its checks to `result.json` and uses a nonzero exit status for a failed check. The optional `runtime-mem` mode with `--private` exercises only the runtime’s allocation, copy and free calls. The automated probe compiles the guest driver from source inside the container.

## Code

| Path | Responsibility |
| --- | --- |
| [guest/driver.c](guest/driver.c) | Implements the supported CUDA Driver calls and models the guest context. |
| [guest/wire.c](guest/wire.c) | Encodes requests and receives the results over a local socket. |
| [host/bridge_host.py](host/bridge_host.py) | Maps guest handles to host resources and dispatches the requested work through CuMetal. |
| [analysis/private_exports.c](analysis/private_exports.c) | Implements the private tables the pinned NVIDIA runtime queries during startup. |
| [guest/probe.c](guest/probe.c) | Verifies the original runtime and the value returned to Linux. |

See the pinned ABI observations and third-party notices for the private compatibility code. [ABI notes](analysis/ABI.md) · [Third-party notices](analysis/NOTICE.txt)

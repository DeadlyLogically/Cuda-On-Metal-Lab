# Reducing driver launch overhead

When CuMetal creates a private JIT function, the generated shader and its static shared-memory size belong to that module until unload. This patch reads and validates the size when the function is created. Later launches can reuse it and skip a lookup in CuMetal’s registered-kernel table. Calls that supply external or mutable function descriptors keep their original lookup behavior. The optimization applies to any eligible kernel. [Runtime and driver changes](fast-launch.patch)

## Recorded measurements

The benchmark submitted a chain of 16 dependent additions over the same 50,000 FP32 values. The M2 ran six pairs of fresh processes in alternating order. The Tesla T4 ran the same CUBIN in one Colab process with interleaved trials. Each chain checked its output and guard values. The times below are microseconds per addition. [Summary](evidence/results.json) · [M2 paired observations](evidence/paired.json) · [T4 observations](evidence/t4.json)

| Measurement | M2 before | M2 after | T4 |
| --- | ---: | ---: | ---: |
| Wait after each of the 16 launches | 220.219 | 215.443 | 13.080 |
| Wait after the full chain | 25.697 | 22.494 | 6.395 |
| Host submission time per launch in the queued chain | 6.897 | 4.136 | — |
| Separate synchronized single-launch test | 179.417 | 206.250 | — |

The synchronized single-launch test varied across processes and its median increased. The existing runtime already supported submitting multiple commands before waiting. The performance data covers the vector-add workload. GPT‑2 training throughput after this patch remains unmeasured.

## Build and compare

Follow the SASS instructions from the sibling directory through the build of the driver with SASS support. Use the same source checkout, compiler and build directory for both variants. The commands below run from the repository root. [SASS build and prerequisites](../sass/README.md)

```sh
cp sass/local/build/libcumetal.dylib sass/local/before-launch.dylib
git -C sass/local/cumetal apply --check "$PWD/launch/fast-launch.patch"
git -C sass/local/cumetal apply "$PWD/launch/fast-launch.patch"
cmake --build sass/local/build --target cumetal_runtime
```

With the same toolchain and disassembler environment as the SASS test, run the dependent-chain probe separately for each library. It reports host submission and synchronization times in addition to wall time and checks every output element.

```sh
export PATH="$PWD/sass/local/build:$PATH"
python3 launch/profile.py --library sass/local/before-launch.dylib --out sass/local/before-chain.json --counts 50000 --batches 16 --trials 9
python3 launch/profile.py --library sass/local/build/libcumetal.dylib --out sass/local/after-chain.json --counts 50000 --batches 16 --trials 9
```

For a full correctness replay, rerun the SASS verifier with the newly built runtime and a fresh output directory.

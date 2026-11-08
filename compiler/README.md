# PTX translation and Warp integration

CuMetal already compiled supported PTX into Metal for Apple GPUs. These additions change how CuMetal chooses a kernel's support code, loads PTX at runtime and passes arguments to the compiled kernel. This directory contains a source overlay for CuMetal, one patch for Warp and small numerical tests that exercise original CUDA kernels from llm.c.

| Workload | Behavior change | Evidence on Apple M2 |
| --- | --- | --- |
| llm.c Softmax | Registration checks the selected kernel and its reachable helpers. An unused double-precision kernel in the same file had previously sent an FP32 kernel down the offline compile path. | [240 cases and 3,352,992 checked outputs](evidence/softmax.json) |
| llm.c Attention | Registration bundles CuMetal's existing FP64 helpers with the generated Metal source for runtime compilation. The original three-kernel attention sequence uses these helpers for its internal double arithmetic. | [72 cases and 216 GPU launches](evidence/attention.json) |
| llm.c LayerNorm | The existing translator already handled the tested kernels. The harness checks that behavior across the runtime change. | [111 cases](evidence/layernorm.json) |
| Warp | The driver compiles the requested PTX entry at runtime and accepts raw PTX modules up to its 64 MiB image limit. The CUDA headers and Warp build also gained support for compiling the original CUB reductions. | [24 selected upstream tests and 892 GPU launches](evidence/warp.json) |

The release overlay was built again from the pinned upstream checkout. Five focused CTest checks, all three CUDA sweeps and the fixed Warp manifest passed on M2. This replay used the existing prepared Warp/CUB native library. [Release verification](evidence/release-check.json)

## Code map

- The registration runtime selects and embeds the helpers. [Runtime registration](overlay/runtime/registration/registration.cpp) · [CMake embedding function](overlay/cmake/EmbedMetalSupport.cmake)
- The CUDA driver loads PTX, compiles the selected entry and binds its arguments. The PTX argument layout code is shared with the command-line compiler. [Driver](overlay/runtime/driver/cuda_driver.cpp) · [Argument layout](overlay/compiler/ptx/src/kernel_abi.cpp)
- The source overlay contains changed CuMetal files in full. Most of each existing file comes from CuMetal. The applier verifies file hashes before updating a pinned checkout. The Warp patch lets a prepared macOS build choose PTX when the CuMetal runtime compiler is active. [Applier](apply.py) · [Warp patch](warp-output.patch)

## Build on Apple Silicon

Use macOS with Xcode command-line tools, CMake 3.28 or newer, Ninja, Python 3, lz4 and zstd. Run the following from the root of this repository. The runtime tests use the system Metal compiler. The CUDA source tests also need a Clang with an NVPTX target. The recorded runs used LLVM Clang 22.

```bash
release="$PWD/compiler"
git clone --recursive https://github.com/Lulzx/cuda-metal.git cumetal-ptx
git -C cumetal-ptx checkout 912694289d5751ff529db94016388744084873b4
git -C cumetal-ptx submodule update --init --recursive
python3 "$release/apply.py" "$PWD/cumetal-ptx" --check
python3 "$release/apply.py" "$PWD/cumetal-ptx"
export CUMETAL_SOURCE_DIR="$PWD/cumetal-ptx"
export CUMETAL_BUILD_DIR="$PWD/build-ptx"
cmake -S "$CUMETAL_SOURCE_DIR" -B "$CUMETAL_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_DISABLE_FIND_PACKAGE_LLVM=ON \
  -DCUMETAL_BUILD_TESTS=ON -DCUMETAL_ENABLE_CUDA_REGISTRATION=ON \
  -DCUMETAL_ENABLE_BINARY_SHIM=OFF
cmake --build "$CUMETAL_BUILD_DIR" --target cumetal cumetalc cumetal_runtime \
  cumetal_registration_entry_support_test cumetal_cuda_host_qualifiers_test
ctest --test-dir "$CUMETAL_BUILD_DIR" --output-on-failure \
  -R '^(functional_registration_entry_support|functional_driver_runtime_msl|unit_embed_metal_support|unit_warp_sweep_harness|unit_cuda_host_qualifiers)$'
```

If CMake cannot find Homebrew dependencies, add `-DCMAKE_PREFIX_PATH="$(brew --prefix)"` during configuration.

## Run the CUDA examples

Set `CUMETAL_CLANG` to the absolute path to your NVPTX-capable `clang++`. Each test reads the pinned original CUDA source from the llm.c Git checkout, compiles a small harness and verifies the results against an independent numerical reference. Each accepted result also requires trace records showing execution on the Apple GPU. Give each run a new directory.

```bash
git clone https://github.com/karpathy/llm.c.git llm.c
export LLMC_SOURCE_DIR="$PWD/llm.c"
export CUMETAL_CLANG="$(brew --prefix llvm)/bin/clang++"
bash "$release/examples/run.sh" softmax sweep "$PWD/softmax-output"
bash "$release/examples/run.sh" layernorm sweep "$PWD/layernorm-output"
bash "$release/examples/run.sh" attention sweep "$PWD/attention-output"
```

## Warp setup

The integration uses Warp v1.12.0 at `e6c3ba2d54bb048115760b5cd7a4bb2573329ae7`. Start with the two Warp build patches from the pinned CuMetal checkout. Then apply this project's PTX output selection patch.

```bash
git clone https://github.com/NVIDIA/warp.git warp-prepared
git -C warp-prepared checkout e6c3ba2d54bb048115760b5cd7a4bb2573329ae7
warp="$PWD/warp-prepared"
bash "$CUMETAL_SOURCE_DIR/scripts/warp-patches/apply_warp_patches.sh" "$warp"
git -C "$warp" apply --unidiff-zero "$release/warp-output.patch"
bash "$CUMETAL_SOURCE_DIR/scripts/build_llama_cpp_cumetal.sh" --toolkit-only
```

Build the Warp native library with its v1.12.0 build dependencies and a Python environment with NumPy. The original experiment obtained the matching `warp/bin/libwarp-clang.dylib` from NVIDIA's official macOS arm64 v1.12.0 wheel. Keep that library in the prepared tree before creating the CUB variant. Capture the native compile and link commands in the build log.

```bash
export SDKROOT="$(xcrun --show-sdk-path)"
(cd "$warp" && python3 build_lib.py --cuda-path="$CUMETAL_BUILD_DIR/cumetal-cuda-toolkit" \
  --no-standalone --no-use-libmathdx --jobs=2 --verbose) > "$PWD/warp-build.log" 2>&1
git clone https://github.com/NVIDIA/cub.git cub
git -C cub checkout c493b3bf143ccd66c917a1982ea64bb2a8a82932
git clone https://github.com/NVIDIA/thrust.git thrust
git -C thrust checkout 1ac51f2b6219ff17d15d93f2e0be85038556f346
python3 "$CUMETAL_SOURCE_DIR/scripts/build_warp_upstream_cub.py" \
  --prepared-warp "$warp" --variant-warp "$PWD/warp-with-cub" \
  --cub "$PWD/cub" --thrust "$PWD/thrust" --build-log "$PWD/warp-build.log" \
  --nvcc "$CUMETAL_BUILD_DIR/cumetal-cuda-toolkit/bin/nvcc" --output "$PWD/cub-build-report"
git -C "$PWD/warp-with-cub" apply --unidiff-zero "$release/warp-output.patch"
python3 "$CUMETAL_SOURCE_DIR/tests/functional/run_warp_sweep.py" \
  --cumetal "$CUMETAL_BUILD_DIR/cumetal" --warp "$PWD/warp-with-cub" \
  --python "$(command -v python3)" --clang "$CUMETAL_CLANG" \
  --output "$PWD/warp-results" --timeout 180
```

These checks cover a fixed manifest of source-built Warp tests. The CUB variant recompiles Warp's original reduction source against the pinned CUB and Thrust headers. The other native objects are checked by hash and reused.

## Attribution

The existing runtime and compiler are derived from CuMetal's pinned Apache-2.0 sources. This project modified the files in the overlay. VF64 support comes from CuMetal's pinned submodule. The CUDA tests load llm.c's original source at commit `f1e2ace651495b74ae22d45d1723443fd00ecd3a`. Warp, CUB and Thrust are external build inputs. [CuMetal license](../licenses/CuMetal-Apache-2.0.txt) · [Warp license](Warp-LICENSE.md) · [llm.c license](../licenses/llm.c-LICENSE)


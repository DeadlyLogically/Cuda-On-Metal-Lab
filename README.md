# Loading Turing binaries on Apple Silicon

This prototype loads standalone NVIDIA CUDA binaries that contain machine instructions for the Tesla T4 (`sm_75`). The added loader sends the original code and parameter layout to a translator. The translator uses NVIDIA’s `nvdisasm`, checks the decoded instructions against their original bytes, and emits PTX for CuMetal’s existing Metal backend. The Mac executes the supported kernels locally.

The tests cover straight-line FP32 arithmetic, per-thread memory access and the bounds checks used by the four positive fixtures. The unsupported atomic reduction fails before a GPU launch. [Translation code](lifter/lift.py) · [Loader patch](mac/driver.patch) · [Test cases](fixtures/README.md)

## Compare the recorded results

The prepared baseline failed to launch the original `vectorAdd` file. With the loader, 36 file-based cases produced the same 207,472 output values on M2 and a physical Colab Tesla T4. Nine additional cases loaded the vector addition image from memory. The negative case records the instruction and offset that stopped translation. [Mac results](verified/comparison.json) · [T4 reference](t4/t4.json)

The saved runs refer to the original build and translator hashes. This package retains their fixture bytes, reformats the scripts and publishes the compiler/runtime portion of the prerequisite patch. The packaged source also passed a fresh M2 replay against the saved T4 outputs. [Packaged replay](verified/packaged-replay.json)

Timing and binaries from a rebuild can differ from the recorded run. The source for the NVIDIA sample stays upstream. Its binary and the modified test binary carry the [original notice](fixtures/NVIDIA-vectorAdd-LICENSE.txt).

## Reproduce on a Mac

Use Apple Silicon with Metal, Xcode command line tools, Python 3.10 or newer, Git, CMake, Ninja and Docker. Set `NVDISASM_BINARY` to a Linux ARM64 `nvdisasm` executable and `NVDISASM_CONTAINER` to a compatible Linux container image available to Docker. The wrapper mounts the tool and the selected CUBIN read-only. The fixtures were disassembled with CUDA 12.8.

Run the following commands from this `sass` directory. The first build supplies the baseline runtime used in the negative control.

```sh
export NVDISASM_BINARY=/absolute/path/to/linux-arm64/nvdisasm
export NVDISASM_CONTAINER=debian:bookworm-slim
export CUMETAL_SASS_NVDISASM="$PWD/mac/nvdisasm-docker"

python3 fixtures/verify.py
python3 lifter/test_lift.py

mkdir -p local
git clone https://github.com/Lulzx/cuda-metal.git local/cumetal
git -C local/cumetal checkout --detach 912694289d5751ff529db94016388744084873b4
git -C local/cumetal submodule update --init --recursive
python3 baseline/apply.py local/cumetal
cmake -S local/cumetal -B local/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_DISABLE_FIND_PACKAGE_LLVM=ON \
  -DCUMETAL_BUILD_TESTS=OFF -DCUMETAL_ENABLE_CUDA_REGISTRATION=ON \
  -DCUMETAL_ENABLE_BINARY_SHIM=OFF
cmake --build local/build --target cumetalc cumetal_runtime
cp local/build/libcumetal.dylib local/baseline.dylib

git -C local/cumetal apply --unidiff-zero "$PWD/mac/driver.patch"
cmake --build local/build --target cumetal_runtime
python3 verify_mac.py --build local/build --baseline-lib local/baseline.dylib \
  --lib local/build/libcumetal.dylib --results local/results \
  --t4 t4/t4.json --t4-bench t4/t4-speed.json
```

The verifier compares output and guard hashes, checks the explicit failure cases, and runs a synchronized single-launch timing. The CMake build uses CuMetal and its pinned VF64-metal submodule. The prerequisites include the runtime PTX path and a parser correction for scalar parameters used in address arithmetic. [Upstream CuMetal license](baseline/CuMetal-LICENSE.txt)

## Reproduce on a Tesla T4

Upload [SASS-T4.ipynb](SASS-T4.ipynb) to Colab and select a T4 runtime. The first cell accepts a ZIP of this repository and locates the necessary files under `sass/`. You can also prepare a small ZIP from the repository root with the command below. The notebook runs the original CUBIN bytes and downloads two JSON files. Its timing covers a single CUDA launch followed by synchronization.

```sh
zip -q t4-input.zip sass/bench.py sass/t4/oracle.py sass/t4/oracle-input.json sass/fixtures/NVIDIA-vectorAdd-LICENSE.txt sass/fixtures/build/*.cubin
```

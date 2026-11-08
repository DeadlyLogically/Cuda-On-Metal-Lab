# Turing test programs

These five standalone CUDA ELF files target `sm_75`. They contain SASS and no PTX. The original NVIDIA vector addition program and a two-byte modification of that binary are distributed with the [source's BSD notice](NVIDIA-vectorAdd-LICENSE.txt). [Pinned NVIDIA CUDA sample](https://github.com/NVIDIA/cuda-samples/blob/0ec046d5e99d7214cc8e9fbac2cc1e66f3019878/cpp/0_Introduction/vectorAdd_nvrtc/vectorAdd_kernel.cu)

| Binary | Expected result |
| --- | --- |
| `vectorAdd.sm75.cubin` | Add two FP32 vectors |
| `elementMultiply.sm75.cubin` | Multiply two FP32 vectors |
| `tripleSum.sm75.cubin` | Add the two inputs, then add the second input once more |
| `sameNameMultiply.sm75.cubin` | Multiply the inputs although the entry point is still named `vectorAdd` |
| `atomicSum.sm75.cubin` | Reject the unsupported atomic reduction before launch |

Run `python3 fixtures/verify.py` from the SASS directory to check the saved binary hashes, disassembly, parameter controls and empty PTX dumps. Instruction offsets displayed as `/*...*/` in the disassembler output are part of its output format and are checked by the binary reader.

To regenerate the fixtures, set `CUDA12_SDK` to the Linux ARM64 CUDA 12.6 SDK directory, `CUDA_BINARY_UTILS` to the directory of the Linux ARM64 CUDA 12.8 `nvdisasm` and `cuobjdump` executables, and `BUILD_IMAGE` to a local ARM64 image with a supported host C++ compiler and Python. Run `bash fixtures/build.sh`. The script retrieves the pinned NVIDIA source and verifies its SHA-256 before compilation. The historical fixture hashes depend on the recorded toolkit versions.

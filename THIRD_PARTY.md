# Source credits

- **CuMetal** provides the existing PTX-to-Metal compiler, runtime and Metal backend. The compatibility experiments build on revision [`9126942`](https://github.com/Lulzx/cuda-metal/tree/912694289d5751ff529db94016388744084873b4). Its license is included under [licenses](licenses/CuMetal-Apache-2.0.txt).
- **NVIDIA CUDA Samples** provides the original vector-add test used for the two-byte control. The fixture documentation identifies the exact source revision. `nvdisasm` and NVRTC are external NVIDIA tools. [Sample source](https://github.com/NVIDIA/cuda-samples/blob/0ec046d5e99d7214cc8e9fbac2cc1e66f3019878/cpp/0_Introduction/vectorAdd_nvrtc/vectorAdd_kernel.cu)
- **ZLUDA** provided a reference for private export-table layouts and a compatibility algorithm used in the Linux bridge. The bridge includes its attribution and the upstream license texts. [Upstream revision](https://github.com/vosen/ZLUDA/tree/f4df215edbc3b6698a1c61ef7a8688fdff3872be/dark_api)
- **llm.c** provides the original training and kernel workloads. The generated Metal benchmark source was derived from its FP32 matrix kernel. [llm.c license](licenses/llm.c-LICENSE) · [Training source](https://github.com/karpathy/llm.c/blob/f1e2ace651495b74ae22d45d1723443fd00ecd3a/train_gpt2_fp32.cu)
- **NVIDIA Warp** provides the upstream test cases exercised by the compiler and runtime changes. Refer to the compiler directory for the tested configuration and scope. [Warp project](https://github.com/NVIDIA/warp)

The component directories contain further notices for the materials they include.

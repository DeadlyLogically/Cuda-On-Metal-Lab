#!/usr/bin/env bash
set -euo pipefail
fixtures="$(cd "$(dirname "$0")" && pwd)"
sdk="${CUDA12_SDK:?Set CUDA12_SDK to the Linux ARM64 toolkit directory}"
utils="${CUDA_BINARY_UTILS:?Set CUDA_BINARY_UTILS to the Linux ARM64 binary utilities directory}"
image="${BUILD_IMAGE:?Set BUILD_IMAGE to a Linux ARM64 image with g++ and Python 3}"
mkdir -p "$fixtures/build"
source_dir="$(mktemp -d)"
trap 'rm -rf "$source_dir"' EXIT
cp "$fixtures"/src/*.cu "$source_dir/"
curl -fsSL --retry 2 https://raw.githubusercontent.com/NVIDIA/cuda-samples/0ec046d5e99d7214cc8e9fbac2cc1e66f3019878/cpp/0_Introduction/vectorAdd_nvrtc/vectorAdd_kernel.cu -o "$source_dir/vectorAdd_kernel.cu"
python3 - "$source_dir/vectorAdd_kernel.cu" <<'CHECK'
from hashlib import sha256
from pathlib import Path
import sys

expected = 'fe190d34eb7f7675d64c3a8589af4a44bc4398aa5a7d616b8d01370f28f8a745'
if sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != expected:
    raise SystemExit("Upstream vectorAdd source hash mismatch")
CHECK
docker run --rm --network none \
    -v "$sdk:/sdk:ro" -v "$utils:/bins:ro" \
    -v "$source_dir:/src:ro" -v "$fixtures/build:/out" \
    -v "$fixtures/byte_control.py:/byte_control.py:ro" \
    "$image" sh -ec '
    export LD_LIBRARY_PATH=/sdk/nvvm/lib64:/sdk/lib64
    /sdk/bin/nvcc --version > /out/nvcc-version.txt
    /bins/nvdisasm --version > /out/nvdisasm-version.txt
    /bins/cuobjdump --version > /out/cuobjdump-version.txt
    for kernel in vectorAdd elementMultiply tripleSum atomicSum; do
        src="$kernel"
        if [ "$kernel" = vectorAdd ]; then src=vectorAdd_kernel; fi
        base=/out/$kernel.sm75
        /sdk/bin/nvcc --cubin -arch=sm_75 -o "$base.cubin" "/src/$src.cu"
        /bins/nvdisasm -c "$base.cubin" > "$base.sass"
        /bins/nvdisasm -c -hex "$base.cubin" > "$base.hex.sass"
        /bins/cuobjdump --dump-ptx "$base.cubin" > "$base.embedded-ptx.txt"
        test ! -s "$base.embedded-ptx.txt"
    done
    python3 /byte_control.py /out
    base=/out/sameNameMultiply.sm75
    /bins/nvdisasm -c "$base.cubin" > "$base.sass"
    /bins/nvdisasm -c -hex "$base.cubin" > "$base.hex.sass"
    /bins/cuobjdump --dump-ptx "$base.cubin" > "$base.embedded-ptx.txt"
    test ! -s "$base.embedded-ptx.txt"
'
python3 "$fixtures/verify.py"

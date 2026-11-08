#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
    echo "usage: run.sh {softmax|layernorm|attention} {smoke|sweep} NEW_OUTPUT_DIRECTORY" >&2
    exit 64
fi

program="$1"
mode="$2"
case "$program" in softmax|layernorm|attention) ;; *) exit 64 ;; esac
case "$mode" in smoke|sweep) ;; *) exit 64 ;; esac
: "${LLMC_SOURCE_DIR:?Set LLMC_SOURCE_DIR to an llm.c Git checkout}"
: "${CUMETAL_SOURCE_DIR:?Set CUMETAL_SOURCE_DIR to the patched CuMetal checkout}"
: "${CUMETAL_BUILD_DIR:?Set CUMETAL_BUILD_DIR to its CMake build}"
: "${CUMETAL_CLANG:?Set CUMETAL_CLANG to the CUDA-capable clang++ executable}"

examples="$(cd "$(dirname "$0")" && pwd)"
mkdir "$3"
export PROBE_DIR
PROBE_DIR="$(cd "$3" && pwd)"
mkdir "$PROBE_DIR/upstream"
ref="f1e2ace651495b74ae22d45d1723443fd00ecd3a"
for file in common.h "${program}_forward.cu"; do
    git -C "$LLMC_SOURCE_DIR" show "$ref:dev/cuda/$file" > "$PROBE_DIR/upstream/$file"
done
cp "$examples/${program}_auto.cu" "$PROBE_DIR"
sdk="$(xcrun --show-sdk-path)"
export CUMETAL_CUDA_PROJECT_EXTRA_DEVICE_FLAGS="-isysroot $sdk -include cuda_runtime.h -include cuda_bf16.h"
source "$CUMETAL_SOURCE_DIR/tests/cuda_projects/_common.sh"
cumetal_cuda_projects_check_prereqs "$CUMETAL_SOURCE_DIR"
cumetal_cuda_projects_compile_link "$CUMETAL_SOURCE_DIR" "$PROBE_DIR" "$PROBE_DIR" "${program}_auto.cu" "${program}_auto"
if [[ "$program" == softmax ]]; then
    python3 "$examples/check_softmax.py" "$mode"
else
    python3 "$examples/check_transformer.py" "$program" "$mode"
fi


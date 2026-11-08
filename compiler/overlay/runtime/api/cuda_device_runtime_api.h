#pragma once

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__clang__) && defined(__CUDA__)
__device__ cudaError_t cudaLaunchDevice(void*, void*, dim3, dim3,
                                        unsigned int, cudaStream_t);
#else
static inline cudaError_t cudaLaunchDevice(void*, void*, dim3, dim3, unsigned int, cudaStream_t) {
    return cudaErrorUnknown;
}
#endif

#ifdef __cplusplus
}
#endif

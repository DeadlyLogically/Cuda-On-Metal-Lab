#include "vector_types.h"
#include "cuda.h"

#ifndef __HOST_DEFINES_H__
#undef __host__
#undef __device__
#endif
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"

#include <cstdio>
#include <type_traits>

using DeviceLaunch = cudaError_t (*)(void*, void*, dim3, dim3, unsigned int, cudaStream_t);
static_assert(std::is_same_v<decltype(&cudaLaunchDevice), DeviceLaunch>,
              "device launches take one packed parameter buffer, not a pointer table");

static int synchronize_calls = 0;
extern "C" cudaError_t cudaDeviceSynchronize(void) {
    ++synchronize_calls;
    return cudaErrorInvalidValue;
}

__host__ __device__ __forceinline__ int host_device_max(int a, int b)
{
    return a > b ? a : b;
}

__global__ void host_parsed_kernel(int* value)
{
    *value = host_device_max(*value, 7);
}

int main()
{
    if (__cudaDeviceSynchronizeDeprecationAvoidance() != cudaErrorInvalidValue ||
        synchronize_calls != 1) {
        std::fprintf(stderr, "FAIL: synchronization compatibility helper lost the runtime result\n");
        return 1;
    }
    unsigned char packed_arguments[16] = {};
    if (cudaLaunchDevice(nullptr, static_cast<void*>(packed_arguments),
                         dim3(1), dim3(1), 0, nullptr) != cudaErrorUnknown) {
        std::fprintf(stderr, "FAIL: host device-launch placeholder must refuse execution\n");
        return 1;
    }
    static_assert(CUDA_VERSION >= 12000, "CUDA_VERSION must select typed wrappers");
    static_assert(CU_EVENT_DISABLE_TIMING == 2, "CUDA event flag ABI mismatch");
    static_assert(CU_JIT_MAX_REGISTERS == 0, "CUDA JIT option ABI mismatch");
    static_assert(CU_AD_FORMAT_FLOAT == 0x20, "CUDA array format ABI mismatch");
    static_assert(CU_TR_FILTER_MODE_LINEAR == 1, "CUDA texture filter ABI mismatch");

    int value = 3;
    host_parsed_kernel(&value);
    if (value != 7) {
        std::fprintf(stderr, "FAIL: host compiler did not parse CUDA qualifiers\n");
        return 1;
    }

    std::printf("PASS: CUDA qualifiers are host-compiler compatible\n");
    return 0;
}

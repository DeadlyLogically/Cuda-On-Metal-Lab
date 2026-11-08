#include "cuda_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

extern "C" {
void** __cudaRegisterFatBinary(const void*);
void __cudaUnregisterFatBinary(void**);
void __cudaRegisterFunction(void**, const void*, char*, const char*, int,
                            void*, void*, void*, void*, int*);
}

namespace {
struct Wrapper {
    std::uint32_t magic = 0x466243b1u;
    std::uint32_t version = 1;
    const void* data = nullptr;
    const void* unknown = nullptr;
};

struct BlobHeader {
    std::uint32_t magic = 0xBA55ED50u;
    std::uint16_t version = 1;
    std::uint16_t header_size = 16;
    std::uint64_t fat_size = 0;
};

constexpr char kPtx[] = R"PTX(
.version 7.0
.target sm_80
.address_size 64

.extern .func (.param .b32 result) __nv_roundf(.param .b32 value);

.visible .entry entry_fp32(
    .param .u64 .ptr .global output,
    .param .u64 .ptr .global input,
    .param .u32 count)
{
    .reg .pred %p;
    .reg .b32 %r<3>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<3>;
    ld.param.u64 %rd0, [output];
    ld.param.u64 %rd1, [input];
    ld.param.u32 %r0, [count];
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p, %r1, %r0;
    @%p bra DONE32;
    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    ld.global.f32 %f0, [%rd4];
    add.f32 %f1, %f0, %f0;
    add.f32 %f2, %f1, 0f3F800000;
    st.global.f32 [%rd3], %f2;
DONE32:
    ret;
}

.visible .entry entry_fp64(
    .param .u64 .ptr .global output,
    .param .u64 .ptr .global input,
    .param .u32 count)
{
    .reg .pred %p;
    .reg .b32 %r<3>;
    .reg .b64 %rd<6>;
    .reg .f64 %fd<2>;
    ld.param.u64 %rd0, [output];
    ld.param.u64 %rd1, [input];
    ld.param.u32 %r0, [count];
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p, %r1, %r0;
    @%p bra DONE64;
    mul.wide.u32 %rd2, %r1, 8;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    ld.global.f64 %fd0, [%rd4];
    add.rn.f64 %fd1, %fd0, %fd0;
    st.global.f64 [%rd3], %fd1;
DONE64:
    ret;
}

.visible .entry entry_libdevice(
    .param .u64 .ptr .global output,
    .param .u64 .ptr .global input,
    .param .u32 count)
{
    .reg .pred %p;
    .reg .b32 %r<3>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<2>;
    .param .b32 arg0;
    .param .b32 result0;
    ld.param.u64 %rd0, [output];
    ld.param.u64 %rd1, [input];
    ld.param.u32 %r0, [count];
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p, %r1, %r0;
    @%p bra DONE_LIB;
    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    ld.global.f32 %f0, [%rd4];
    st.param.f32 [arg0], %f0;
    call.uni (result0), __nv_roundf, (arg0);
    ld.param.f32 %f1, [result0];
    st.global.f32 [%rd3], %f1;
DONE_LIB:
    ret;
}

.visible .entry entry_unsupported(
    .param .u64 .ptr .global output,
    .param .u64 .ptr .global input,
    .param .u32 count)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd;
    ld.param.u64 %rd, [output];
    mov.u32 %r0, %tid.x;
    add.cc.u32 %r1, %r0, 1;
    addc.u32 %r2, %r1, 0;
    st.global.u32 [%rd], %r2;
    ret;
}
)PTX";

void host_stub() {}

bool require(cudaError_t status, const char* action) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "FAIL: %s: %s\n", action, cudaGetErrorString(status));
    return false;
}

template<class T>
bool run_numeric(const std::string& kind) {
    constexpr std::uint32_t count = 37;
    constexpr std::size_t guard = 16;
    constexpr T sentinel = T(-777.25);
    std::vector<T> input(count + 2 * guard, sentinel);
    std::vector<T> output(input.size(), sentinel);
    for (std::uint32_t i = 0; i < count; ++i) {
        input[guard + i] = (T(i) - T(18)) * T(0.25);
        output[guard + i] = std::numeric_limits<T>::quiet_NaN();
        if constexpr (std::is_same_v<T, double>) {
            if (i % 2) input[guard + i] += std::ldexp(1.0, -40);
        }
    }
    if constexpr (std::is_same_v<T, double>) {
        const char* mode = std::getenv("CUMETAL_FP64_MODE");
        const bool wide_range = mode && (std::strcmp(mode, "wide48") == 0 ||
                                         std::strcmp(mode, "ieee64") == 0);
        input[guard] = std::ldexp(1.0, wide_range ? 500 : 80);
        input[guard + 1] = std::ldexp(1.0, wide_range ? -500 : -80);
        if (mode && std::strcmp(mode, "ieee64") == 0)
            input[guard + 2] = 1.0 + std::ldexp(1.0, -52);
    }
    const auto bytes = input.size() * sizeof(T);
    void* device_input = nullptr;
    void* device_output = nullptr;
    if (!require(cudaMalloc(&device_input, bytes), "allocate input") ||
        !require(cudaMalloc(&device_output, bytes), "allocate output") ||
        !require(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice), "upload input") ||
        !require(cudaMemcpy(device_output, output.data(), bytes, cudaMemcpyHostToDevice), "upload output")) return false;
    void* input_arg = static_cast<T*>(device_input) + guard;
    void* output_arg = static_cast<T*>(device_output) + guard;
    std::uint32_t count_arg = count;
    void* args[] = {&output_arg, &input_arg, &count_arg};
    const auto launch = cudaLaunchKernel(reinterpret_cast<const void*>(&host_stub),
                                         dim3(1), dim3(64), args, 0, nullptr);
    const auto sync = cudaDeviceSynchronize();
    if (kind == "unsupported") {
        cudaFree(device_input);
        cudaFree(device_output);
        if (launch == cudaSuccess) {
            std::fprintf(stderr, "FAIL: unsupported PTX unexpectedly launched\n");
            return false;
        }
        std::puts("ENTRY_SUPPORT_UNSUPPORTED_REFUSED");
        return true;
    }
    if (!require(launch, "launch") || !require(sync, "synchronization") ||
        !require(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost), "read output")) return false;
    std::vector<T> unchanged(input.size());
    if (!require(cudaMemcpy(unchanged.data(), device_input, bytes, cudaMemcpyDeviceToHost), "read input")) return false;
    if (!require(cudaFree(device_input), "free input") ||
        !require(cudaFree(device_output), "free output")) return false;
    if (std::memcmp(unchanged.data(), input.data(), bytes) != 0) {
        std::fprintf(stderr, "FAIL: input changed\n");
        return false;
    }
    for (std::size_t i = 0; i < output.size(); ++i) {
        T expected = sentinel;
        if (i >= guard && i < guard + count) {
            expected = kind == "libdevice" ? T(std::round(input[i]))
                       : kind == "fp64" ? input[i] + input[i]
                       : input[i] * T(2) + T(1);
        }
        if (output[i] != expected) {
            std::fprintf(stderr, "FAIL: output/guard %zu: %.17g != %.17g\n",
                         i, double(output[i]), double(expected));
            return false;
        }
    }
    std::printf("ENTRY_SUPPORT_NUMERICAL_OK kind=%s values=37\n", kind.c_str());
    return true;
}
}   

int main(int argc, char** argv) {
    if (argc != 2) return 64;
    const std::string kind = argv[1];
    if (kind != "fp32" && kind != "fp64" && kind != "libdevice" &&
        kind != "unsupported") return 64;
    int devices = 0;
    const auto device_status = cudaGetDeviceCount(&devices);
    if (device_status == cudaErrorNoDevice ||
        (device_status == cudaSuccess && devices == 0)) return 77;
    if (!require(device_status, "device count") ||
        !require(cudaSetDevice(0), "set device")) return 1;

    BlobHeader header;
    header.fat_size = sizeof(kPtx) - 1;
    std::vector<std::uint8_t> blob(sizeof(header) + header.fat_size);
    std::memcpy(blob.data(), &header, sizeof(header));
    std::memcpy(blob.data() + sizeof(header), kPtx, header.fat_size);
    Wrapper wrapper;
    wrapper.data = blob.data();
    void** handle = __cudaRegisterFatBinary(&wrapper);
    if (handle == nullptr) {
        std::fprintf(stderr, "FAIL: module registration\n");
        return 1;
    }
    std::string entry = "entry_" + kind;
    __cudaRegisterFunction(handle, reinterpret_cast<const void*>(&host_stub),
                           entry.data(), nullptr, 0, nullptr, nullptr,
                           nullptr, nullptr, nullptr);

    const bool ok = kind == "fp64" ? run_numeric<double>(kind) : run_numeric<float>(kind);
    __cudaUnregisterFatBinary(handle);
    return ok ? 0 : 1;
}

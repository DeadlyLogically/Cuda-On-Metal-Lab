#include "cuda.h"

#include "cumetal/air_emitter/emitter.h"
#include "cumetal/common/metallib.h"
#include "cumetal/metal/lower_to_msl.h"
#include "cumetal/ptx/kernel_abi.h"
#include "cumetal/ptx/lower_to_llvm.h"
#include "cumetal/ptx/lower_to_metal.h"
#include "cumetal/ptx/parser.h"
#include "metal_math_mode.h"
#include "cumetal_diag.h"
#include "cuda_runtime.h"
#include "fatbin_elf.h"
#include "fatbin_ptx.h"
#include "metal_backend.h"
#include "module_cache.h"
#include "metal_support_sources.h"

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct CUctx_st {
    CUdevice device = 0;
    unsigned int flags = 0;
};

struct CUfunc_st;

struct DriverGlobalSymbol {
    std::string name;
    std::uint32_t size = 0;
    std::uint32_t alignment = 1;
    std::vector<std::uint8_t> initializer;
};

struct DriverPtxModule {
    std::string source;
    std::vector<std::string> entries;
    std::string fp64_mode;
    std::string math_mode;
    std::filesystem::path workspace;
    std::map<std::string, std::string> artifacts;
    std::mutex mutex;

    ~DriverPtxModule() {
        if (!workspace.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(workspace, ec);
        }
    }
};

struct CUmod_st {
    std::string metallib_path;
    bool owns_metallib_path = false;
    std::shared_ptr<DriverPtxModule> ptx;
    std::vector<CUfunc_st*> functions;

    std::map<std::string, std::pair<void*, std::uint32_t>> global_storage;
};

struct CUfunc_st {
    CUmod_st* module = nullptr;
    std::string kernel_name;
    std::string runtime_msl_path;
    std::uint32_t argument_count = 0;
    bool has_argument_count = false;
    std::vector<cumetalKernelArgInfo_t> argument_info;
    std::vector<DriverGlobalSymbol> global_symbols;
};

namespace {

const std::string& function_module_path(const CUfunc_st* function) {
    return function->runtime_msl_path.empty() ? function->module->metallib_path
                                             : function->runtime_msl_path;
}

extern "C" int cumetalRuntimeIsDevicePointer(const void* ptr);
extern "C" cudaError_t cumetalFuncAttributesFromKernelProperties(
    const cumetal::metal_backend::KernelProperties& kernel, cudaFuncAttributes* attr);
extern "C" int cumetalRuntimeGetAllocationInfo(const void* ptr, void** base_out, size_t* size_out);
extern "C" int cumetalRuntimeIsManaged(const void* ptr);
extern "C" void* cumetalRuntimeGetHostPointer(const void* ptr, size_t count);

constexpr int kCudaCompatVersion = 12000;
constexpr std::uint32_t kFatbinWrapperMagic = 0x466243b1u;
constexpr std::size_t kMaxImageBytes = 64ull * 1024ull * 1024ull;

struct FatbinWrapper {
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    const void* data = nullptr;
    const void* unknown = nullptr;
};

struct DriverState {
    struct VmmHandleRecord {
        std::size_t size = 0;
        CUmemAllocationProp prop{};
    };
    struct VmmReservationRecord {
        std::size_t size = 0;
        bool mapped = false;
        bool access_enabled = false;
        CUmemAllocationProp mapped_prop{};
    };

    std::mutex mutex;
    bool initialized = false;
    std::unordered_set<CUctx_st*> contexts;
    std::unordered_set<CUmod_st*> modules;
    std::unordered_set<CUfunc_st*> functions;
    std::unordered_map<CUmemGenericAllocationHandle, VmmHandleRecord> vmm_handles;
    std::unordered_map<CUdeviceptr, VmmReservationRecord> vmm_reservations;
    CUmemGenericAllocationHandle next_vmm_handle = 1;
    CUctx_st* runtime_primary_context = nullptr;
    unsigned int primary_ctx_flags = 0;
};

thread_local CUctx_st* g_current_context = nullptr;
thread_local std::vector<CUctx_st*> g_context_stack;

struct DriverStreamCallbackPayload {
    CUstream stream = nullptr;
    CUstreamCallback callback = nullptr;
    void* user_data = nullptr;
};

DriverState& driver_state() {

    static DriverState* state = new DriverState();
    return *state;
}

CUresult map_cuda_error(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return CUDA_SUCCESS;
        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevicePointer:
            return CUDA_ERROR_INVALID_VALUE;
        case cudaErrorMemoryAllocation:
            return CUDA_ERROR_OUT_OF_MEMORY;
        case cudaErrorInitializationError:
            return CUDA_ERROR_NOT_INITIALIZED;
        case cudaErrorLaunchTimeout:
            return CUDA_ERROR_LAUNCH_TIMEOUT;
        case cudaErrorCooperativeLaunchTooLarge:
            return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        case cudaErrorNotReady:
            return CUDA_ERROR_NOT_READY;
        case cudaErrorIllegalAddress:
            return CUDA_ERROR_ILLEGAL_ADDRESS;
        case cudaErrorDevicesUnavailable:
            return CUDA_ERROR_DEVICES_UNAVAILABLE;

        case cudaErrorCallRequiresNewerDriver:
        case cudaErrorOperatingSystem:
        case cudaErrorNotSupported:
            return CUDA_ERROR_NOT_SUPPORTED;
        case cudaErrorCudartUnloading:
            return CUDA_ERROR_DEINITIALIZED;
        case cudaErrorNoDevice:
            return CUDA_ERROR_NO_DEVICE;
        case cudaErrorInvalidDevice:
            return CUDA_ERROR_INVALID_DEVICE;
        case cudaErrorInvalidDeviceFunction:
            return CUDA_ERROR_NOT_FOUND;
        case cudaErrorInvalidConfiguration:
        case cudaErrorInvalidMemcpyDirection:
            return CUDA_ERROR_INVALID_VALUE;
        case cudaErrorInvalidResourceHandle:
            return CUDA_ERROR_INVALID_HANDLE;
        case cudaErrorLaunchOutOfResources:
            return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        case cudaErrorLaunchFailure:
            return CUDA_ERROR_LAUNCH_FAILED;
        case cudaErrorAssert:
            return CUDA_ERROR_ASSERT;
        case cudaErrorNotPermitted:
            return CUDA_ERROR_NOT_PERMITTED;
        case cudaErrorInsufficientDriver:
            return CUDA_ERROR_NOT_SUPPORTED;
        case cudaErrorPeerAccessAlreadyEnabled:
            return CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
        case cudaErrorPeerAccessNotEnabled:
            return CUDA_ERROR_PEER_ACCESS_NOT_ENABLED;
        case cudaErrorGraphExecUpdateFailure:
            return CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE;
        case cudaErrorUnknown:
        default:

            return CUDA_ERROR_UNKNOWN;
    }
}

bool has_current_context_locked(const DriverState& state) {
    return g_current_context != nullptr &&
           state.contexts.find(g_current_context) != state.contexts.end();
}

CUresult require_initialized_context() {
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (!has_current_context_locked(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    return CUDA_SUCCESS;
}

bool is_valid_function_locked(const DriverState& state, CUfunction function) {
    return function != nullptr && state.functions.find(function) != state.functions.end();
}

CUresult query_function_properties(
    CUfunction function,
    cumetal::metal_backend::KernelProperties* properties) {
    if (properties == nullptr) return CUDA_ERROR_INVALID_VALUE;
    std::string metallib_path;
    std::string kernel_name;
    {
        DriverState& state = driver_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) return CUDA_ERROR_NOT_INITIALIZED;
        if (!has_current_context_locked(state)) return CUDA_ERROR_INVALID_CONTEXT;
        if (!is_valid_function_locked(state, function) || function->module == nullptr)
            return CUDA_ERROR_INVALID_VALUE;
        metallib_path = function_module_path(function);
        kernel_name = function->kernel_name;
    }
    std::string error;
    return map_cuda_error(cumetal::metal_backend::query_kernel_properties(
        metallib_path, kernel_name, properties, &error));
}

bool parse_unsigned(const std::string& text, unsigned long long limit,
                    unsigned long long* out) {
    if (text.empty() || text.size() > 20 ||
        !std::all_of(text.begin(), text.end(),
                     [](unsigned char c) { return std::isdigit(c) != 0; })) {
        return false;
    }
    const unsigned long long value = std::strtoull(text.c_str(), nullptr, 10);
    if (value > limit) return false;
    *out = value;
    return true;
}

struct SidecarKernel {
    std::string name;
    std::vector<cumetalKernelArgInfo_t> info;
    std::vector<DriverGlobalSymbol> globals;
};

bool parse_abi_sidecar(const std::string& metallib_path, std::vector<SidecarKernel>* kernels) {
    kernels->clear();
    const std::filesystem::path abi_path = metallib_path + ".cumetal-abi";
    std::vector<std::string> tokens;
    {
        std::ifstream abi(abi_path);
        std::string header;
        if (!abi || !std::getline(abi, header) ||
            (header != "CUMETAL_ABI_V1" && header != "CUMETAL_ABI_V2")) {
            return false;
        }
        for (std::string token; abi >> token;) {
            if (tokens.size() >= 4096) {
                return false;
            }
            tokens.push_back(std::move(token));
        }
    }

    for (std::size_t i = 0; i + 1 < tokens.size();) {
        if (tokens[i] != "kernel") return false;
        SidecarKernel kernel;
        kernel.name = tokens[i + 1];
        i += 2;

        bool valid = true;
        for (; i < tokens.size() && tokens[i] != "kernel";) {
            if (tokens[i] == "shared") {
                unsigned long long shared_bytes = 0;
                if (i + 1 >= tokens.size() ||
                    !parse_unsigned(tokens[i + 1], 16ull * 1024ull * 1024ull, &shared_bytes)) {
                    valid = false;
                    break;
                }
                i += 2;
                continue;
            }
            if (tokens[i] == "global") {
                 
                unsigned long long size = 0;
                unsigned long long alignment = 0;
                if (i + 4 >= tokens.size() ||
                    !parse_unsigned(tokens[i + 2], 64ull * 1024ull, &size) || size == 0 ||
                    !parse_unsigned(tokens[i + 3], 256, &alignment) || alignment == 0) {
                    valid = false;
                    break;
                }
                DriverGlobalSymbol symbol{
                    .name = tokens[i + 1],
                    .size = static_cast<std::uint32_t>(size),
                    .alignment = static_cast<std::uint32_t>(alignment),
                };
                const std::string& encoded = tokens[i + 4];
                if (encoded != "-") {
                    if (encoded.size() != size * 2) {
                        valid = false;
                        break;
                    }
                    const auto nibble = [](char value) -> int {
                        if (value >= '0' && value <= '9') return value - '0';
                        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
                        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
                        return -1;
                    };
                    symbol.initializer.reserve(size);
                    for (std::size_t byte = 0; byte < encoded.size(); byte += 2) {
                        const int high = nibble(encoded[byte]);
                        const int low = nibble(encoded[byte + 1]);
                        if (high < 0 || low < 0) {
                            valid = false;
                            break;
                        }
                        symbol.initializer.push_back(
                            static_cast<std::uint8_t>((high << 4) | low));
                    }
                    if (!valid) break;
                }
                kernel.globals.push_back(std::move(symbol));
                i += 5;
                continue;
            }
            unsigned long long size = 0;
            if (tokens[i] != "arg" || i + 2 >= tokens.size() ||
                (tokens[i + 1] != "buffer" && tokens[i + 1] != "bytes") ||
                !parse_unsigned(tokens[i + 2], 4096, &size) || size == 0 ||
                kernel.info.size() >= 31) {
                valid = false;
                break;
            }
            kernel.info.push_back(cumetalKernelArgInfo_t{
                .kind = tokens[i + 1] == "buffer" ? CUMETAL_ARG_BUFFER : CUMETAL_ARG_BYTES,
                .size_bytes = static_cast<std::uint32_t>(size),
            });
            i += 3;
        }
        if (!valid) return false;
        kernels->push_back(std::move(kernel));
    }
    return true;
}

void load_function_argument_count(CUfunc_st* function) {
    if (function == nullptr || function->module == nullptr) {
        return;
    }

    std::vector<SidecarKernel> kernels;
    parse_abi_sidecar(function_module_path(function), &kernels);
    for (SidecarKernel& kernel : kernels) {
        if (kernel.name == function->kernel_name) {
            function->argument_count = static_cast<std::uint32_t>(kernel.info.size());
            function->has_argument_count = true;
            function->argument_info = std::move(kernel.info);
            function->global_symbols = std::move(kernel.globals);
            return;
        }
    }

    std::string error;
    const auto bytes =
        cumetal::common::read_file_bytes(function_module_path(function), &error);
    if (bytes.empty()) {
        return;
    }
    const auto summary = cumetal::common::inspect_metallib_bytes(
        function_module_path(function), bytes, 0);
    for (const auto& kernel : summary.kernels) {
        if (kernel.name != function->kernel_name) {
            continue;
        }
        for (const auto& field : kernel.metadata) {
            if (field.key != "kernel.arg_count") {
                continue;
            }
            char* end = nullptr;
            const unsigned long count = std::strtoul(field.value.c_str(), &end, 10);
            if (end != field.value.c_str() && *end == '\0' && count <= 31) {
                function->argument_count = static_cast<std::uint32_t>(count);
                function->has_argument_count = true;
            }
            return;
        }
    }
}

bool is_valid_module_locked(const DriverState& state, CUmodule module) {
    return module != nullptr && state.modules.find(module) != state.modules.end();
}

CUresult create_module_from_path(const std::string& path, bool owns_path, CUmodule* out_module) {
    if (out_module == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!std::filesystem::exists(path)) {
        return CUDA_ERROR_INVALID_IMAGE;
    }

    auto* loaded = new (std::nothrow) CUmod_st{};
    if (loaded == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    loaded->metallib_path = path;
    loaded->owns_metallib_path = owns_path;

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.modules.insert(loaded);
    }

    *out_module = loaded;
    return CUDA_SUCCESS;
}

bool parse_metallib_size(const void* image, std::size_t* out_size) {
    if (image == nullptr || out_size == nullptr) {
        return false;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(image);
    if (std::memcmp(bytes, "MTLB", 4) != 0) {
        return false;
    }

    std::uint64_t raw_size = 0;
    std::memcpy(&raw_size, bytes + 0x10, sizeof(raw_size));
    if (raw_size < 0x20 || raw_size > (1ull << 31)) {
        return false;
    }

    *out_size = static_cast<std::size_t>(raw_size);
    return true;
}

bool parse_metallib_path_image(const void* image, std::string* out_path) {
    if (image == nullptr || out_path == nullptr) {
        return false;
    }

    const auto* chars = static_cast<const char*>(image);
    constexpr std::size_t kMaxPathBytes = 4096;
    std::size_t len = 0;
    for (; len < kMaxPathBytes; ++len) {
        if (chars[len] == '\0') {
            break;
        }
    }
    if (len == 0 || len == kMaxPathBytes) {
        return false;
    }

    std::error_code ec;
    std::filesystem::path candidate(std::string(chars, len));
    if (!std::filesystem::exists(candidate, ec) || ec) {
        return false;
    }
    if (!std::filesystem::is_regular_file(candidate, ec) || ec) {
        return false;
    }

    *out_path = candidate.string();
    return true;
}

bool extract_ptx_cstr(const char* chars, std::size_t max_bytes, std::string* out_ptx) {
    if (chars == nullptr || out_ptx == nullptr || max_bytes == 0) {
        return false;
    }

    const void* terminator = std::memchr(chars, '\0', max_bytes);
    if (terminator == nullptr) {
        return false;
    }

    const std::size_t size = static_cast<const char*>(terminator) - chars;
    if (size == 0) {
        return false;
    }

    const std::string candidate(chars, size);
    if (candidate.find(".version") == std::string::npos ||
        candidate.find(".entry") == std::string::npos) {
        return false;
    }

    *out_ptx = candidate;
    return true;
}

bool parse_direct_ptx_image(const void* image, std::string* out_ptx) {
    if (image == nullptr || out_ptx == nullptr) {
        return false;
    }

    const char* chars = static_cast<const char*>(image);
    constexpr std::size_t kMaxPreamble = 64 * 1024;
    std::size_t i = 0;
    while (i < kMaxPreamble) {
        const char c = chars[i];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            ++i;
        } else if (c == '/' && chars[i + 1] == '/') {
            while (i < kMaxPreamble && chars[i] != '\n' && chars[i] != '\0') ++i;
        } else if (c == '/' && chars[i + 1] == '*') {
            i += 2;
            while (i + 1 < kMaxPreamble && chars[i] != '\0' &&
                   !(chars[i] == '*' && chars[i + 1] == '/')) {
                ++i;
            }
            if (chars[i] == '\0' || i + 1 >= kMaxPreamble) return false;
            i += 2;
        } else {
            break;
        }
    }
    if (i >= kMaxPreamble || chars[i] != '.') {
        return false;
    }

    return extract_ptx_cstr(chars, kMaxImageBytes, out_ptx);
}

bool parse_fatbin_blob_ptx(const void* image, std::string* out_ptx) {
    return cumetal::fatbin::extract_fatbin_ptx(
               image, kMaxImageBytes, out_ptx) ==
           cumetal::fatbin::PtxExtractStatus::kFound;
}

bool parse_fatbin_wrapper_ptx(const void* image, std::string* out_ptx) {
    if (image == nullptr || out_ptx == nullptr) {
        return false;
    }

    const auto* raw = static_cast<const std::uint8_t*>(image);
    constexpr std::size_t kOffsets[] = {0u, 16u};
    for (const std::size_t offset : kOffsets) {
        std::uint32_t magic = 0;
        std::uint32_t version = 0;
        std::memcpy(&magic, raw + offset, sizeof(magic));
        std::memcpy(&version, raw + offset + sizeof(magic), sizeof(version));
        if (magic != kFatbinWrapperMagic || version == 0 || version > 3) {
            continue;
        }

        const void* data = nullptr;
        std::memcpy(&data, raw + offset + sizeof(magic) + sizeof(version), sizeof(data));
        if (data == nullptr) {
            continue;
        }

        if (parse_direct_ptx_image(data, out_ptx)) {
            return true;
        }
        if (parse_fatbin_blob_ptx(data, out_ptx)) {
            return true;
        }
        if (cumetal::fatbin::extract_elf_ptx(
                data, kMaxImageBytes, out_ptx) ==
            cumetal::fatbin::ElfPtxStatus::kFound) {
            return true;
        }
    }
    return false;
}

bool parse_ptx_image(const void* image, std::string* out_ptx) {
    if (image == nullptr || out_ptx == nullptr) {
        return false;
    }

    if (parse_direct_ptx_image(image, out_ptx)) {
        return true;
    }
    if (parse_fatbin_blob_ptx(image, out_ptx)) {
        return true;
    }
    if (parse_fatbin_wrapper_ptx(image, out_ptx)) {
        return true;
    }
    return cumetal::fatbin::extract_elf_ptx(
               image, kMaxImageBytes, out_ptx) ==
           cumetal::fatbin::ElfPtxStatus::kFound;
}

bool runtime_msl_driver_jit_enabled() {
    const char* value = std::getenv("CUMETAL_DRIVER_JIT");
    return value != nullptr && std::string_view(value) == "runtime-msl";
}

CUresult create_runtime_ptx_module(const std::string& source, CUmodule* out_module) {
    const char* backend = std::getenv("CUMETAL_PTX_BACKEND");
     
    if (backend != nullptr && backend[0] != '\0' &&
        std::string_view(backend) != "cumetal-ir") {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    const auto parsed = cumetal::ptx::parse_ptx(source, {.strict = false});
    if (!parsed.ok || parsed.module.entries.empty()) return CUDA_ERROR_INVALID_IMAGE;
    auto module = std::make_unique<CUmod_st>();
    module->ptx = std::make_shared<DriverPtxModule>();
    module->ptx->source = source;
    module->ptx->fp64_mode = std::string(cumetal::ptx::fp64_mode_name(
        cumetal::ptx::fp64_mode_from_env()));
    module->ptx->math_mode =
        cumetal::metal_math_mode_name(cumetal::current_metal_math_mode());
    for (const auto& entry : parsed.module.entries) {
        if (std::find(module->ptx->entries.begin(), module->ptx->entries.end(), entry.name) !=
            module->ptx->entries.end()) return CUDA_ERROR_INVALID_IMAGE;
        module->ptx->entries.push_back(entry.name);
    }
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.modules.insert(module.get());
    *out_module = module.release();
    return CUDA_SUCCESS;
}

CUresult compile_runtime_ptx_entry(const std::shared_ptr<DriverPtxModule>& module,
                                   const std::string& name, std::string* out_path) {
    std::lock_guard<std::mutex> lock(module->mutex);
    if (const auto found = module->artifacts.find(name); found != module->artifacts.end()) {
        *out_path = found->second;
        return CUDA_SUCCESS;
    }
    if (std::find(module->entries.begin(), module->entries.end(), name) ==
        module->entries.end()) return CUDA_ERROR_NOT_FOUND;

    cumetal::metal::PtxToMslOptions options;
    options.strict = true;
    options.entry_name = name;
    options.fp64_mode = module->fp64_mode;
    options.source_name = "driver-runtime-ptx";
    const auto compiled = cumetal::metal::compile_ptx_to_msl(module->source, options);
    const auto report = [&](const std::string& error) {
        if (cumetal::diag_env_truthy("CUMETAL_DEBUG_ERRORS")) {
            std::fprintf(stderr, "CUMETAL_DEBUG_ERRORS: runtime-MSL Driver JIT '%s': %s\n",
                         name.c_str(), error.c_str());
        }
    };
    if (!compiled.ok || compiled.source.empty()) {
        report(compiled.error.empty() ? "no MSL produced" : compiled.error);
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    for (const auto& warning : compiled.warnings) {
        cumetal::warn_once("driver-runtime-msl:" + name + ":" + warning, warning);
    }
    if (!compiled.printf_formats.empty() || compiled.uses_device_heap ||
        compiled.uses_device_launch_queue) {
        report("device printf, device heap, and dynamic launch require registration-owned resources");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    for (const auto& function : compiled.gpu_ir.functions) {
        if (!function.is_kernel || !function.kernel_abi.has_value()) continue;
        for (const auto& binding : function.kernel_abi->bindings) {
            if (binding.hidden_role.has_value() &&
                !binding.hidden_role->starts_with("global_symbol:")) {
                report("unsupported hidden Driver JIT binding: " + *binding.hidden_role);
                return CUDA_ERROR_NOT_SUPPORTED;
            }
        }
    }
    std::string abi_diagnostics;
    const std::string abi = cumetal::ptx::build_ir_abi_sidecar(
        compiled.gpu_ir, name, module->source, &abi_diagnostics);
    if (abi.empty() || !abi_diagnostics.empty()) {
        report(abi_diagnostics.empty() ? "typed kernel ABI missing" : abi_diagnostics);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    const bool needs_fp64 = compiled.source.find("cm_fp64_") != std::string::npos ||
                            compiled.source.find("vf64_") != std::string::npos;
    const bool needs_libdevice = compiled.source.find("cm_libdevice_") != std::string::npos;
    std::string source;
    if (needs_fp64) {
        std::string provenance = "generic_ptx_lowering";
        if (module->fp64_mode == "fast48") provenance += "_fp64_emulated";
        else if (module->fp64_mode == "wide48") provenance += "_fp64_wide48";
        else if (module->fp64_mode == "ieee64") provenance += "_fp64_ieee64";
        source += "// cumetal-provenance: " + provenance + "\n";
    }
    source += "// cumetal-math-mode: " +
              std::string(needs_fp64 ? "safe" : module->math_mode) + "\n";
    if (needs_fp64) {
        source += cumetal::registration::support::kFp64;
        source += '\n';
    }
    if (needs_libdevice) {
        source += cumetal::registration::support::kLibdevice;
        source += '\n';
    }
    source += compiled.source;
    if (std::find(compiled.gpu_ir.semantic_caveats.begin(),
                  compiled.gpu_ir.semantic_caveats.end(),
                  cumetal::ir::kFp64LibdeviceViaF32Caveat) !=
        compiled.gpu_ir.semantic_caveats.end()) {
        cumetal::warn_fp64_libdevice_via_f32(module->fp64_mode);
    }

    if (module->workspace.empty()) {
        std::error_code ec;
        const auto temporary = std::filesystem::temp_directory_path(ec);
        if (ec) return CUDA_ERROR_UNKNOWN;
        const std::string pattern = (temporary / "cumetal-driver-msl-XXXXXX").string();
        std::vector<char> directory(pattern.begin(), pattern.end());
        directory.push_back('\0');
        if (::mkdtemp(directory.data()) == nullptr) return CUDA_ERROR_UNKNOWN;
        module->workspace = directory.data();
    }
    const auto path = module->workspace /
        ("entry-" + std::to_string(module->artifacts.size()) + ".metal");
    const auto abi_path = path.string() + ".cumetal-abi";
    std::string error;
    if (!cumetal::common::write_file_bytes(path, {source.begin(), source.end()}, &error) ||
        !cumetal::common::write_file_bytes(abi_path, {abi.begin(), abi.end()}, &error)) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        std::filesystem::remove(abi_path, ec);
        report(error);
        return CUDA_ERROR_UNKNOWN;
    }

    std::vector<SidecarKernel> kernels;
    if (!parse_abi_sidecar(path.string(), &kernels) || kernels.size() != 1 ||
        kernels.front().name != name) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        std::filesystem::remove(abi_path, ec);
        report("generated ABI failed validation");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    module->artifacts.emplace(name, path.string());
    *out_path = path.string();
    return CUDA_SUCCESS;
}

bool try_typed_ptx_to_temp_metallib(const std::string& ptx, std::string* out_path,
                                    bool* toolchain_missing) {
    const char* backend_env = std::getenv("CUMETAL_PTX_BACKEND");
    const std::string backend = backend_env == nullptr ? "" : backend_env;
    if (backend == "legacy") return false;

    const cumetal::ptx::ParseResult parsed = cumetal::ptx::parse_ptx(ptx);
    if (!parsed.ok || parsed.module.entries.empty()) return false;

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path tmp = std::filesystem::temp_directory_path();
    std::vector<std::filesystem::path> metal_paths;
    const auto cleanup_metal = [&]() {
        std::error_code ec;
        for (const auto& path : metal_paths) std::filesystem::remove(path, ec);
    };

    cumetal::ptx::LowerToMetalOptions options;
    options.backend = cumetal::ptx::PtxMetalBackend::kCumetalIr;
    options.allow_legacy_fallback = false;
    options.fp64_mode = std::string(
        cumetal::ptx::fp64_mode_name(cumetal::ptx::fp64_mode_from_env()));
    for (std::size_t i = 0; i < parsed.module.entries.size(); ++i) {
        options.entry_name = parsed.module.entries[i].name;
        const cumetal::ptx::LowerToMetalResult lowered =
            cumetal::ptx::lower_ptx_to_metal_source(ptx, options);
        if (!lowered.ok || lowered.metal_source.empty() || lowered.approximate) {
            if (cumetal::diag_env_truthy("CUMETAL_DEBUG_ERRORS")) {
                std::fprintf(stderr,
                             "CUMETAL_DEBUG_ERRORS: typed PTX lowering declined "
                             "'%s': %s\n",
                             options.entry_name.c_str(),
                             lowered.error.empty() ? "no MSL produced"
                                                   : lowered.error.c_str());
            }
            cleanup_metal();
            return false;
        }
        if (lowered.metal_source.find("cm_fp64_") != std::string::npos ||
            lowered.metal_source.find("cm_libdevice_") != std::string::npos ||
            lowered.metal_source.find("vf64_") != std::string::npos) {
            cleanup_metal();
            return false;
        }
        const std::filesystem::path metal_path =
            tmp / ("cumetal-driver-typed-" + std::to_string(stamp) + "-" +
                   std::to_string(i) + ".metal");
        std::string io_error;
        const std::vector<std::uint8_t> metal_bytes(lowered.metal_source.begin(),
                                                    lowered.metal_source.end());
        if (!cumetal::common::write_file_bytes(metal_path, metal_bytes, &io_error)) {
            cleanup_metal();
            return false;
        }
        metal_paths.push_back(metal_path);
    }

    const std::filesystem::path metallib_path =
        tmp / ("cumetal-driver-typed-" + std::to_string(stamp) + ".metallib");
    cumetal::air_emitter::EmitOptions emit_options;
    emit_options.input = metal_paths.front();
    for (std::size_t i = 1; i < metal_paths.size(); ++i) {
        emit_options.additional_link_inputs.push_back(metal_paths[i]);
    }
    emit_options.output = metallib_path;
    emit_options.mode = cumetal::air_emitter::EmitMode::kXcrun;
    emit_options.overwrite = true;
    emit_options.validate_output = true;
    emit_options.fallback_to_experimental = true;
    emit_options.kernel_name = parsed.module.entries.front().name;
    emit_options.math_mode =
        cumetal::metal_math_mode_name(cumetal::current_metal_math_mode());
    const cumetal::air_emitter::EmitResult emitted =
        cumetal::air_emitter::emit_metallib(emit_options);
    cleanup_metal();
    if (!emitted.ok || emitted.output.empty()) {
        if (toolchain_missing != nullptr) *toolchain_missing = true;
        return false;
    }
    if (emitted.mode_used ==
        cumetal::air_emitter::EmitMode::kExperimentalContainer) {
        std::error_code ec;
        std::filesystem::remove(emitted.output, ec);
        if (toolchain_missing != nullptr) *toolchain_missing = true;
        return false;
    }

    {
        std::string sidecar = "CUMETAL_ABI_V2\n";
        for (const auto& entry : parsed.module.entries) {
            sidecar += "kernel " + entry.name + "\n";
            sidecar += "shared " +
                       std::to_string(cumetal::ptx::compute_static_shared_bytes(
                           ptx, entry.name)) +
                       "\n";
            for (const auto& param : entry.params) {
                sidecar += param.is_pointer
                               ? "arg buffer 8\n"
                               : "arg bytes " +
                                     std::to_string(param.byte_size == 0
                                                        ? std::size_t{8}
                                                        : param.byte_size) +
                                     "\n";
            }
        }
        const std::vector<std::uint8_t> sidecar_bytes(sidecar.begin(),
                                                      sidecar.end());
        std::string sidecar_error;
        if (!cumetal::common::write_file_bytes(
                emitted.output.string() + ".cumetal-abi", sidecar_bytes,
                &sidecar_error)) {
            std::error_code ec;
            std::filesystem::remove(emitted.output, ec);
            return false;
        }
    }
    *out_path = emitted.output.string();
    return true;
}

bool emit_ptx_to_temp_metallib(const std::string& ptx, std::string* out_path,
                               bool* toolchain_missing) {
    if (ptx.empty() || out_path == nullptr) {
        return false;
    }
    if (try_typed_ptx_to_temp_metallib(ptx, out_path, toolchain_missing)) {
        return true;
    }
    const char* backend_env = std::getenv("CUMETAL_PTX_BACKEND");
    if (backend_env != nullptr && std::string(backend_env) == "cumetal-ir") {

        return false;
    }

    cumetal::ptx::LowerToLlvmOptions lower_opts;
    lower_opts.fp64_mode = cumetal::ptx::fp64_mode_from_env();

    if (lower_opts.fp64_mode == cumetal::ptx::Fp64Mode::kEmulate &&
        ptx.find(".f64") != std::string::npos) {
        cumetal::warn_once(
            "fp64-emulate",
            "kernel uses FP64 (double) instructions, emulated with Dekker FP32-pair "
            "arithmetic (~48-bit significand with binary32 exponent range, not full "
            "precision. Set CUMETAL_FP64_MODE=native to compile true doubles (fails "
            "at launch on current Apple Silicon)");
    }
    const auto lowered = cumetal::ptx::lower_ptx_to_llvm_ir(ptx, lower_opts);
    if (!lowered.ok || lowered.llvm_ir.empty()) {
        if (cumetal::diag_env_truthy("CUMETAL_DEBUG_ERRORS")) {
            std::fprintf(stderr,
                         "CUMETAL_DEBUG_ERRORS: PTX lowering failed: %s\n",
                         lowered.error.empty() ? "empty LLVM IR" : lowered.error.c_str());
        }
        return false;
    }
    if (lowered.fp64_libdevice_via_f32) {
        cumetal::warn_fp64_libdevice_via_f32(cumetal::ptx::fp64_mode_name(lower_opts.fp64_mode));
    }

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path ll_path =
        std::filesystem::temp_directory_path() / ("cumetal-driver-" + std::to_string(stamp) + ".ll");
    const std::filesystem::path metallib_path =
        std::filesystem::temp_directory_path() / ("cumetal-driver-" + std::to_string(stamp) + ".metallib");

    std::string io_error;
    const std::vector<std::uint8_t> ll_bytes(lowered.llvm_ir.begin(), lowered.llvm_ir.end());
    if (!cumetal::common::write_file_bytes(ll_path, ll_bytes, &io_error)) {
        if (cumetal::diag_env_truthy("CUMETAL_DEBUG_ERRORS")) {
            std::fprintf(stderr,
                         "CUMETAL_DEBUG_ERRORS: failed to stage PTX LLVM IR: %s\n",
                         io_error.c_str());
        }
        return false;
    }

    cumetal::air_emitter::EmitOptions emit_options;
    emit_options.input = ll_path;
    emit_options.output = metallib_path;
    emit_options.mode = cumetal::air_emitter::EmitMode::kXcrun;
    emit_options.overwrite = true;
    emit_options.validate_output = false;
    emit_options.fallback_to_experimental = true;
    emit_options.kernel_name = lowered.entry_name.empty() ? "vector_add" : lowered.entry_name;
    if ((ptx.find(".f64") != std::string::npos &&
         cumetal::ptx::fp64_mode_links_vf64_support(lower_opts.fp64_mode)) ||
        lowered.llvm_ir.find("vf64_") != std::string::npos) {
        emit_options.additional_link_inputs.push_back(
            std::filesystem::path(CUMETAL_SOURCE_DIR) / "compiler" /
            "metal" / "support" / "cumetal_fp64_support.metal"
        );
    }
    if (lowered.llvm_ir.find("cm_libdevice_") != std::string::npos) {
        emit_options.additional_link_inputs.push_back(
            std::filesystem::path(CUMETAL_SOURCE_DIR) / "compiler" /
            "metal" / "support" / "cumetal_libdevice_support.metal"
        );
    }

    const auto emitted = cumetal::air_emitter::emit_metallib(emit_options);
    std::error_code ec;
    std::filesystem::remove(ll_path, ec);
    if (!emitted.ok || emitted.output.empty()) {
        if (cumetal::diag_env_truthy("CUMETAL_DEBUG_ERRORS")) {
            std::fprintf(stderr,
                         "CUMETAL_DEBUG_ERRORS: PTX metallib emission failed: %s\n",
                         emitted.error.empty() ? "empty output path" : emitted.error.c_str());
        }
        std::filesystem::remove(metallib_path, ec);
        return false;
    }
    if (emitted.mode_used == cumetal::air_emitter::EmitMode::kExperimentalContainer) {

        cumetal::warn_once(
            "metal-toolchain-missing",
            "kernel JIT needs Apple's offline Metal compiler (`xcrun metal`), which was "
            "not found; kernels that require it cannot run. Install Xcode, then run "
            "`xcodebuild -downloadComponent MetalToolchain`.");
        std::filesystem::remove(emitted.output, ec);
        *toolchain_missing = true;
        return false;
    }

    *out_path = emitted.output.string();
    return true;
}

bool stage_module_image_to_tempfile(const void* image, std::size_t size, std::string* out_path) {
    if (image == nullptr || size == 0 || out_path == nullptr) {
        return false;
    }

    const std::string filename =
        "cumetal_module_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".metallib";
    const std::filesystem::path path = std::filesystem::temp_directory_path() / filename;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    out.write(static_cast<const char*>(image), static_cast<std::streamsize>(size));
    out.close();
    if (!out.good()) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        return false;
    }

    *out_path = path.string();
    return true;
}

bool valid_context_flags(unsigned int flags) {
    constexpr unsigned int kSupportedContextFlags =
        CU_CTX_SCHED_SPIN | CU_CTX_SCHED_YIELD | CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST |
        CU_CTX_LMEM_RESIZE_TO_MAX;

    if ((flags & ~kSupportedContextFlags) != 0) {
        return false;
    }

    const unsigned int schedule_bits =
        flags & (CU_CTX_SCHED_SPIN | CU_CTX_SCHED_YIELD | CU_CTX_SCHED_BLOCKING_SYNC);
    if (schedule_bits == (CU_CTX_SCHED_SPIN | CU_CTX_SCHED_YIELD) ||
        schedule_bits == (CU_CTX_SCHED_SPIN | CU_CTX_SCHED_BLOCKING_SYNC) ||
        schedule_bits == (CU_CTX_SCHED_YIELD | CU_CTX_SCHED_BLOCKING_SYNC) ||
        schedule_bits == (CU_CTX_SCHED_SPIN | CU_CTX_SCHED_YIELD | CU_CTX_SCHED_BLOCKING_SYNC)) {
        return false;
    }

    return true;
}

void driver_stream_callback_bridge(cudaStream_t stream, cudaError_t status, void* user_data) {
    (void)stream;
    auto* payload = static_cast<DriverStreamCallbackPayload*>(user_data);
    if (payload == nullptr || payload->callback == nullptr) {
        return;
    }

    payload->callback(payload->stream, map_cuda_error(status), payload->user_data);
    delete payload;
}

bool is_runtime_device_pointer_word(CUdeviceptr value) {
    const std::uintptr_t raw = static_cast<std::uintptr_t>(value);
    if (raw == 0) {
        return false;
    }
    return cumetalRuntimeIsDevicePointer(reinterpret_cast<const void*>(raw)) != 0;
}

constexpr std::size_t kDriverVmmGranularity = 64u * 1024u;

bool is_power_of_two(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool valid_vmm_prop(const CUmemAllocationProp& prop) {
    return prop.type == CU_MEM_ALLOCATION_TYPE_PINNED &&
           prop.requestedHandleTypes == CU_MEM_HANDLE_TYPE_NONE &&
           prop.location.type == CU_MEM_LOCATION_TYPE_DEVICE &&
           prop.location.id == 0 && prop.win32HandleMetaData == nullptr &&
           (prop.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_NONE ||
            prop.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC) &&
           prop.allocFlags.gpuDirectRDMACapable == 0 && prop.allocFlags.usage == 0;
}

}   

extern "C" {

int cumetalDriverRuntimeActivatePrimaryContext(int device) {
    if (device != 0) return 0;
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.initialized = true;
    if (state.runtime_primary_context == nullptr) {
        auto* context = new (std::nothrow) CUctx_st{};
        if (context == nullptr) return 0;
        context->device = device;
        context->flags = state.primary_ctx_flags;
        state.contexts.insert(context);
        state.runtime_primary_context = context;
    }
    if (g_current_context == nullptr) {
        g_current_context = state.runtime_primary_context;
    }
    return 1;
}

CUresult cuInit(unsigned int flags) {
    if (flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const CUresult status = map_cuda_error(cudaInit(0));
    if (status != CUDA_SUCCESS) {
        return status;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.initialized = true;
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion,
                          uint64_t  , CUdriverProcAddressQueryResult* symbolStatus) {
    if (symbolStatus != nullptr) *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    if (symbol == nullptr || pfn == nullptr || cudaVersion < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pfn = nullptr;
    static void* const self_handle = [] {
        Dl_info info{};
        if (dladdr(reinterpret_cast<const void*>(&cuGetProcAddress), &info) == 0 ||
            info.dli_fname == nullptr) {
            return static_cast<void*>(nullptr);
        }
        return dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    }();
    if (self_handle == nullptr) return CUDA_ERROR_NOT_FOUND;

    if (std::strncmp(symbol, "cu", 2) != 0 || symbol[2] < 'A' || symbol[2] > 'Z') {
        return CUDA_ERROR_NOT_FOUND;
    }
    void* resolved = dlsym(self_handle, symbol);
    if (resolved == nullptr) return CUDA_ERROR_NOT_FOUND;
    *pfn = resolved;
    if (symbolStatus != nullptr) *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int* driverVersion) {
    if (driverVersion == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *driverVersion = kCudaCompatVersion;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int* count) {
    if (count == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    *count = 1;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (ordinal != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    *device = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    if (name == nullptr || len <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    cudaDeviceProp prop{};
    const cudaError_t status = cudaGetDeviceProperties(&prop, dev);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    std::strncpy(name, prop.name, static_cast<std::size_t>(len - 1));
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
    if (bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    cudaDeviceProp prop{};
    const cudaError_t status = cudaGetDeviceProperties(&prop, dev);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    *bytes = prop.totalGlobalMem;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
    if (pi == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    cudaDeviceProp prop{};
    const cudaError_t status = cudaGetDeviceProperties(&prop, dev);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    switch (attrib) {
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            *pi = prop.maxThreadsPerBlock;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
            *pi = prop.maxThreadsDim[0];
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
            *pi = prop.maxThreadsDim[1];
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
            *pi = prop.maxThreadsDim[2];
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
            *pi = prop.maxGridSize[0];
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
            *pi = prop.maxGridSize[1];
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
            *pi = prop.maxGridSize[2];
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:

        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN:
            *pi = static_cast<int>(prop.sharedMemPerBlock);
            break;

        case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
        case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
        case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED:
            *pi = 1;   
            break;
        case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
            *pi = prop.warpSize;
            break;
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
            *pi = prop.multiProcessorCount;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
            *pi = 65536;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR:
            *pi = prop.regsPerMultiprocessor;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR:
            *pi = prop.maxBlocksPerMultiProcessor;
            break;
        case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
            *pi = 1296000;   
            break;
        case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:
            *pi = 1;   
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
            *pi = 8;   
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
            *pi = 0;   
            break;
        case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
        case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:
        case CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED:
        case CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:

            *pi = 0;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }

    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    if (pctx == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!valid_context_flags(flags)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    auto* context = new (std::nothrow) CUctx_st{};
    if (context == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    context->device = dev;
    context->flags = flags;

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.contexts.insert(context);
        if (g_current_context == nullptr) {
            g_current_context = context;
        }
    }

    *pctx = context;
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    if (ctx == nullptr) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.contexts.find(ctx) == state.contexts.end()) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        state.contexts.erase(ctx);
        if (state.runtime_primary_context == ctx) {
            state.runtime_primary_context = nullptr;
        }
        if (g_current_context == ctx) {
            g_current_context = nullptr;
            g_context_stack.clear();
        }
    }

    delete ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (ctx != nullptr && state.contexts.find(ctx) == state.contexts.end()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    g_current_context = ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
    if (pctx == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    *pctx = has_current_context_locked(state) ? g_current_context : nullptr;
    return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent(CUcontext ctx) {
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (ctx == nullptr || state.contexts.find(ctx) == state.contexts.end()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    g_context_stack.push_back(g_current_context);
    g_current_context = ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext* pctx) {
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (!has_current_context_locked(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (pctx != nullptr) {
        *pctx = g_current_context;
    }
    g_current_context = g_context_stack.empty() ? nullptr : g_context_stack.back();
    if (!g_context_stack.empty()) {
        g_context_stack.pop_back();
    }
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    if (pctx == nullptr) return CUDA_ERROR_INVALID_VALUE;
    return cuCtxCreate(pctx, 0, dev);
}

CUresult cuDevicePrimaryCtxRelease(CUdevice  ) {
     
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    if (ctx != nullptr) {
        cuCtxDestroy(ctx);
    }
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
    if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (flags != nullptr) *flags = state.primary_ctx_flags;
    if (active != nullptr) *active = state.contexts.empty() ? 0 : 1;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.primary_ctx_flags = flags;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
    if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
     
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    if (ctx != nullptr) {
        cuCtxDestroy(ctx);
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev) {
    if (uuid == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;

    static const unsigned char kUuid[16] = {
        0x43, 0x75, 0x4d, 0x65, 0x74, 0x61, 0x6c, 0x31,   
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01
    };
    for (int i = 0; i < 16; ++i) uuid->bytes[i] = kUuid[i];
    return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice* device) {
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (!has_current_context_locked(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    *device = g_current_context->device;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetFlags(unsigned int* flags) {
    if (flags == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (!has_current_context_locked(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    *flags = g_current_context->flags;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetFlags(unsigned int flags) {
    if (!valid_context_flags(flags)) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (!has_current_context_locked(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    g_current_context->flags = flags;
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
    }

    return map_cuda_error(cudaDeviceSynchronize());
}

CUresult cuStreamCreate(CUstream* phStream, unsigned int flags) {
    if (flags != CU_STREAM_DEFAULT && flags != CU_STREAM_NON_BLOCKING) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaStreamCreateWithFlags(reinterpret_cast<cudaStream_t*>(phStream), flags));
}

CUresult cuStreamDestroy(CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuStreamSynchronize(CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuStreamQuery(CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaStreamQuery(reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuStreamGetPriority(CUstream  , int* priority) {
    if (priority == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *priority = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags) {
    if (flags == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(
        cudaStreamGetFlags(reinterpret_cast<cudaStream_t>(hStream), flags));
}

CUresult cuStreamGetCtx(CUstream  , CUcontext* pctx) {
    if (pctx == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (!has_current_context_locked(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    *pctx = g_current_context;
    return CUDA_SUCCESS;
}

CUresult cuStreamAddCallback(CUstream hStream,
                             CUstreamCallback callback,
                             void* userData,
                             unsigned int flags) {
    if (callback == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }

    auto* payload = new (std::nothrow) DriverStreamCallbackPayload{};
    if (payload == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    payload->stream = hStream;
    payload->callback = callback;
    payload->user_data = userData;

    const cudaError_t status = cudaStreamAddCallback(reinterpret_cast<cudaStream_t>(hStream),
                                                     driver_stream_callback_bridge,
                                                     payload,
                                                     flags);
    if (status != cudaSuccess) {
        delete payload;
        return map_cuda_error(status);
    }

    return CUDA_SUCCESS;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(hStream),
                                              reinterpret_cast<cudaEvent_t>(hEvent),
                                              flags));
}

CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) {
    if (fn == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
     
    struct Payload {
        CUhostFn fn;
        void* userData;
    };
    auto* payload = new Payload{fn, userData};
    auto wrapper = [](cudaStream_t, cudaError_t, void* arg) {
        auto* p = static_cast<Payload*>(arg);
        p->fn(p->userData);
        delete p;
    };
    cudaError_t status = cudaStreamAddCallback(reinterpret_cast<cudaStream_t>(hStream),
                                               wrapper, payload, 0);
    if (status != cudaSuccess) {
        delete payload;
        return map_cuda_error(status);
    }
    return CUDA_SUCCESS;
}

CUresult cuEventCreate(CUevent* phEvent, unsigned int flags) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t*>(phEvent), flags));
}

CUresult cuEventDestroy(CUevent hEvent) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(hEvent)));
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventRecord(reinterpret_cast<cudaEvent_t>(hEvent),
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventRecordWithFlags(reinterpret_cast<cudaEvent_t>(hEvent),
                                                   reinterpret_cast<cudaStream_t>(hStream),
                                                   flags));
}

CUresult cuEventSynchronize(CUevent hEvent) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(hEvent)));
}

CUresult cuEventQuery(CUevent hEvent) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventQuery(reinterpret_cast<cudaEvent_t>(hEvent)));
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventElapsedTime(pMilliseconds,
                                               reinterpret_cast<cudaEvent_t>(hStart),
                                               reinterpret_cast<cudaEvent_t>(hEnd)));
}

CUresult ensure_module_global_storage(CUmod_st* module,
                                      const DriverGlobalSymbol& symbol,
                                      void** out_storage) {
    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!is_valid_module_locked(state, module)) return CUDA_ERROR_INVALID_HANDLE;
        const auto existing = module->global_storage.find(symbol.name);
        if (existing != module->global_storage.end()) {
            if (existing->second.second != symbol.size) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            *out_storage = existing->second.first;
            return CUDA_SUCCESS;
        }
    }

    void* storage = nullptr;
    if (cudaMalloc(&storage, symbol.size) != cudaSuccess || storage == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    const cudaError_t seeded =
        symbol.initializer.size() == symbol.size
            ? cudaMemcpy(storage, symbol.initializer.data(), symbol.size,
                         cudaMemcpyHostToDevice)
            : cudaMemset(storage, 0, symbol.size);
    if (seeded != cudaSuccess) {
        cudaFree(storage);
        return map_cuda_error(seeded);
    }

    void* surplus = nullptr;
    CUresult published = CUDA_SUCCESS;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!is_valid_module_locked(state, module)) {
            surplus = storage;
            published = CUDA_ERROR_INVALID_HANDLE;
        } else {
            const auto [entry, inserted] = module->global_storage.emplace(
                symbol.name, std::make_pair(storage, symbol.size));
             
            if (!inserted) surplus = storage;
            *out_storage = entry->second.first;
        }
    }
    if (surplus != nullptr) cudaFree(surplus);
    return published;
}

CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes,
                            CUmodule hmod, const char* name) {
    if (name == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (dptr)  *dptr  = 0;
    if (bytes) *bytes = 0;
    if (hmod == nullptr) return CUDA_ERROR_INVALID_VALUE;
    std::shared_ptr<DriverPtxModule> ptx;
    std::string module_path;
    std::vector<DriverGlobalSymbol> known_symbols;
    {
        DriverState& state = driver_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!is_valid_module_locked(state, hmod)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        ptx = hmod->ptx;
        module_path = hmod->metallib_path;
        for (const auto* function : hmod->functions) {
            known_symbols.insert(known_symbols.end(), function->global_symbols.begin(),
                                 function->global_symbols.end());
        }
    }
    const auto resolve = [&](const DriverGlobalSymbol& symbol) -> CUresult {
        {
            DriverState& state = driver_state();
            std::lock_guard<std::mutex> lock(state.mutex);
            if (!is_valid_module_locked(state, hmod) || hmod->ptx != ptx)
                return CUDA_ERROR_INVALID_HANDLE;
        }
        void* storage = nullptr;
        const CUresult ready = ensure_module_global_storage(hmod, symbol, &storage);
        if (ready != CUDA_SUCCESS) return ready;
        if (dptr) {
            *dptr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(storage));
        }
        if (bytes) *bytes = symbol.size;
        return CUDA_SUCCESS;
    };
    for (const auto& symbol : known_symbols) {
        if (symbol.name == name) return resolve(symbol);
    }

    std::vector<SidecarKernel> kernels;
    parse_abi_sidecar(module_path, &kernels);
    for (const SidecarKernel& kernel : kernels) {
        for (const DriverGlobalSymbol& symbol : kernel.globals) {
            if (symbol.name == name) return resolve(symbol);
        }
    }
    if (ptx) {

        for (const auto& entry : ptx->entries) {
            std::string path;
            if (compile_runtime_ptx_entry(ptx, entry, &path) != CUDA_SUCCESS) continue;
            kernels.clear();
            if (!parse_abi_sidecar(path, &kernels)) continue;
            for (const auto& kernel : kernels) {
                for (const auto& symbol : kernel.globals) {
                    if (symbol.name == name) return resolve(symbol);
                }
            }
        }
    }
    return CUDA_ERROR_NOT_FOUND;
}

CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    if (module == nullptr || fname == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
    }

    if (runtime_msl_driver_jit_enabled() && std::filesystem::path(fname).extension() == ".ptx") {
        std::string error;
        const auto bytes = cumetal::common::read_file_bytes(fname, &error);
        if (bytes.empty() || bytes.size() > kMaxImageBytes) return CUDA_ERROR_INVALID_IMAGE;
        return create_runtime_ptx_module(std::string(bytes.begin(), bytes.end()), module);
    }
    return create_module_from_path(fname,  false, module);
}

struct CUlinkState_st {
    std::vector<char> image;
    bool has_input = false;
};

CUresult cuLinkCreate(unsigned int  , CUjit_option*  ,
                      void**  , CUlinkState* stateOut) {
    if (stateOut == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *stateOut = new (std::nothrow) CUlinkState_st();
    return *stateOut != nullptr ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size,
                       const char*  , unsigned int  ,
                       CUjit_option*  , void**  ) {
    if (state == nullptr || data == nullptr || size == 0) return CUDA_ERROR_INVALID_VALUE;
    if (type != CU_JIT_INPUT_CUBIN && type != CU_JIT_INPUT_PTX &&
        type != CU_JIT_INPUT_FATBINARY) {
         
        return type >= CU_JIT_NUM_INPUT_TYPES ? CUDA_ERROR_INVALID_VALUE
                                              : CUDA_ERROR_NOT_SUPPORTED;
    }

    if (state->has_input) return CUDA_ERROR_NOT_SUPPORTED;
    const char* bytes = static_cast<const char*>(data);
    state->image.assign(bytes, bytes + size);
     
    if (type == CU_JIT_INPUT_PTX && state->image.back() != '\0') state->image.push_back('\0');
    state->has_input = true;
    return CUDA_SUCCESS;
}

CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path,
                       unsigned int numOptions, CUjit_option* options, void** optionValues) {
    if (state == nullptr || path == nullptr) return CUDA_ERROR_INVALID_VALUE;
    std::ifstream file(path, std::ios::binary);
    if (!file) return CUDA_ERROR_FILE_NOT_FOUND;
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    if (bytes.empty()) return CUDA_ERROR_INVALID_IMAGE;
    return cuLinkAddData(state, type, bytes.data(), bytes.size(), path, numOptions, options,
                         optionValues);
}

CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) {
    if (state == nullptr || cubinOut == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!state->has_input) return CUDA_ERROR_INVALID_VALUE;
    *cubinOut = state->image.data();
    if (sizeOut != nullptr) *sizeOut = state->image.size();
    return CUDA_SUCCESS;
}

CUresult cuLinkDestroy(CUlinkState state) {
    if (state == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    delete state;
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    if (module == nullptr || image == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
    }

    cumetal::cache::ModuleImageParts parts;
    if (cumetal::cache::parse_module_image(image, kMaxImageBytes, &parts)) {
        std::filesystem::path cache_path;
        std::string cache_error;
        if (!cumetal::cache::stage_metallib_bytes(parts.metallib, parts.metallib_size,
                                                  &cache_path, &cache_error)) {
            return CUDA_ERROR_UNKNOWN;
        }

        cumetal::cache::stage_metallib_abi_sidecar(cache_path, parts.sidecar,
                                                   parts.sidecar_size, &cache_error);
        return create_module_from_path(cache_path.string(),  false, module);
    }

    std::string ptx_text;
    if (parse_ptx_image(image, &ptx_text)) {
        if (runtime_msl_driver_jit_enabled()) {
            return create_runtime_ptx_module(ptx_text, module);
        }
        std::string compiled_metallib_path;
        bool toolchain_missing = false;
        if (emit_ptx_to_temp_metallib(ptx_text, &compiled_metallib_path, &toolchain_missing)) {
            const CUresult load_status =
                create_module_from_path(compiled_metallib_path,  true, module);
            if (load_status != CUDA_SUCCESS) {
                std::error_code ec;
                std::filesystem::remove(compiled_metallib_path, ec);
            }
            return load_status;
        }
        if (toolchain_missing) {
            return CUDA_ERROR_JIT_COMPILER_NOT_FOUND;
        }
    }

    std::size_t size = 0;
    if (parse_metallib_size(image, &size)) {
        std::filesystem::path cache_path;
        if (cumetal::cache::stage_metallib_bytes(image, size, &cache_path, nullptr)) {
            return create_module_from_path(cache_path.string(),  false, module);
        }

        std::string staged_path;
        if (!stage_module_image_to_tempfile(image, size, &staged_path)) {
            return CUDA_ERROR_UNKNOWN;
        }

        const CUresult load_status = create_module_from_path(staged_path,  true, module);
        if (load_status != CUDA_SUCCESS) {
            std::error_code ec;
            std::filesystem::remove(staged_path, ec);
        }
        return load_status;
    }

    std::string image_path;
    if (parse_metallib_path_image(image, &image_path)) {
        return create_module_from_path(image_path,  false, module);
    }

    return CUDA_ERROR_INVALID_IMAGE;
}

CUresult cuModuleLoadDataEx(CUmodule* module,
                            const void* image,
                            unsigned int numOptions,
                            void* options,
                            void* optionValues) {
    if ((numOptions > 0) && (options == nullptr || optionValues == nullptr)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuModuleLoadData(module, image);
}

CUresult cuModuleUnload(CUmodule module) {
    if (module == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }

    DriverState& state = driver_state();
    std::string owned_path;
    bool remove_owned_path = false;
    std::vector<void*> released_globals;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!is_valid_module_locked(state, module)) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        for (CUfunc_st* function : module->functions) {
            state.functions.erase(function);
            delete function;
        }
        module->functions.clear();
        for (const auto& [name, storage] : module->global_storage) {
            (void)name;
            released_globals.push_back(storage.first);
        }
        module->global_storage.clear();
        if (module->owns_metallib_path) {
            owned_path = module->metallib_path;
            remove_owned_path = true;
        }
        state.modules.erase(module);
    }

    for (void* storage : released_globals) cudaFree(storage);

    delete module;
    if (remove_owned_path) {
        std::error_code ec;
        std::filesystem::remove(owned_path, ec);
    }
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    if (hfunc == nullptr || hmod == nullptr || name == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::shared_ptr<DriverPtxModule> ptx;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (!is_valid_module_locked(state, hmod)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        ptx = hmod->ptx;
    }

    auto function = std::unique_ptr<CUfunc_st>(new (std::nothrow) CUfunc_st{});
    if (!function) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (ptx) {
        const auto result = compile_runtime_ptx_entry(ptx, name, &function->runtime_msl_path);
        if (result != CUDA_SUCCESS) return result;
    }
    function->module = hmod;
    function->kernel_name = name;

    {
        std::lock_guard<std::mutex> lock(state.mutex);

        if (!is_valid_module_locked(state, hmod) || hmod->ptx != ptx)
            return CUDA_ERROR_INVALID_HANDLE;
        load_function_argument_count(function.get());
        if (ptx && !function->has_argument_count) return CUDA_ERROR_NOT_SUPPORTED;
        hmod->functions.push_back(function.get());
        state.functions.insert(function.get());
    }

    *hfunc = function.release();
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams,
                          void** extra) {
    if (config == nullptr || (config->numAttrs != 0 && config->attrs == nullptr)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (unsigned int i = 0; i < config->numAttrs; ++i) {
        if (config->attrs[i].id != CU_LAUNCH_ATTRIBUTE_IGNORE) return CUDA_ERROR_NOT_SUPPORTED;
    }
    return cuLaunchKernel(f, config->gridDimX, config->gridDimY, config->gridDimZ,
                          config->blockDimX, config->blockDimY, config->blockDimZ,
                          config->sharedMemBytes, config->hStream, kernelParams, extra);
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,
                        unsigned int gridDimY,
                        unsigned int gridDimZ,
                        unsigned int blockDimX,
                        unsigned int blockDimY,
                        unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void** kernelParams,
                        void** extra) {
    if (f == nullptr || gridDimX == 0 || gridDimY == 0 || gridDimZ == 0 || blockDimX == 0 ||
        blockDimY == 0 || blockDimZ == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (kernelParams != nullptr && extra != nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (!is_valid_function_locked(state, f)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }

    std::vector<void*> launch_params;
    std::vector<CUdeviceptr> packed_arg_values;
    std::vector<cumetalKernelArgInfo_t> arg_info;

    if (extra != nullptr) {

        if (!f->runtime_msl_path.empty()) return CUDA_ERROR_NOT_SUPPORTED;
        void* packed_buffer = nullptr;
        std::size_t packed_size = 0;
        bool have_buffer = false;
        bool have_size = false;

        for (std::size_t i = 0;;) {
            void* key = extra[i++];
            if (key == CU_LAUNCH_PARAM_END) {
                break;
            }
            if (key == CU_LAUNCH_PARAM_BUFFER_POINTER) {
                if (extra[i] == nullptr) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                packed_buffer = extra[i++];
                have_buffer = true;
                continue;
            }
            if (key == CU_LAUNCH_PARAM_BUFFER_SIZE) {
                if (extra[i] == nullptr) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                packed_size = *reinterpret_cast<const std::size_t*>(extra[i]);
                have_size = true;
                ++i;
                continue;
            }
            return CUDA_ERROR_INVALID_VALUE;
        }

        if (!have_buffer || !have_size || (packed_size % sizeof(CUdeviceptr) != 0)) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        const std::size_t arg_count = packed_size / sizeof(CUdeviceptr);
        if (arg_count > 31) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        const auto* packed_args = static_cast<const CUdeviceptr*>(packed_buffer);
        packed_arg_values.assign(packed_args, packed_args + arg_count);
        launch_params.reserve(arg_count);
        arg_info.reserve(arg_count);
        for (std::size_t i = 0; i < packed_arg_values.size(); ++i) {
            launch_params.push_back(&packed_arg_values[i]);
            if (is_runtime_device_pointer_word(packed_arg_values[i])) {
                arg_info.push_back(cumetalKernelArgInfo_t{
                    .kind = CUMETAL_ARG_BUFFER,
                    .size_bytes = 0,
                });
            } else {
                const std::uint32_t scalar_size =
                    (packed_arg_values[i] <= 0xFFFFFFFFull) ? 4u : 8u;
                arg_info.push_back(cumetalKernelArgInfo_t{
                    .kind = CUMETAL_ARG_BYTES,
                    .size_bytes = scalar_size,
                });
            }
        }
    } else if (kernelParams != nullptr) {
        std::size_t arg_count = 0;
        if (f->has_argument_count) {
            arg_count = f->argument_count;
        } else {

            cumetal::warn_once(
                "driver_launch_without_abi_metadata",
                "cuLaunchKernel is guessing the argument count by scanning "
                "kernelParams for a NULL terminator, which CUDA does not "
                "guarantee. The metallib has no .cumetal-abi sidecar.");
            for (; arg_count < 31; ++arg_count) {
                if (kernelParams[arg_count] == nullptr) {
                    break;
                }
            }
            if (arg_count == 31 && kernelParams[31] != nullptr) {
                return CUDA_ERROR_INVALID_VALUE;
            }
        }

        launch_params.reserve(arg_count);
        arg_info.reserve(arg_count);
        for (std::size_t i = 0; i < arg_count; ++i) {
            if (reinterpret_cast<std::uintptr_t>(kernelParams[i]) < 4096) {
                std::fprintf(stderr,
                             "CuMetal Driver API: kernel '%s' argument %zu of %zu has invalid address %p\n",
                             f->kernel_name.c_str(), i, arg_count, kernelParams[i]);
                return CUDA_ERROR_INVALID_VALUE;
            }
            launch_params.push_back(kernelParams[i]);
            if (f->argument_info.size() == arg_count) {
                arg_info.push_back(f->argument_info[i]);
            } else {
                std::uintptr_t value = 0;
                std::memcpy(&value, kernelParams[i], sizeof(value));
                const bool is_pointer = is_runtime_device_pointer_word(value);
                arg_info.push_back(cumetalKernelArgInfo_t{
                    .kind = is_pointer ? CUMETAL_ARG_BUFFER : CUMETAL_ARG_BYTES,
                    .size_bytes = is_pointer ? 0u : (value <= 0xFFFFFFFFull ? 4u : 8u),
                });
            }
        }
    }

    std::vector<std::uintptr_t> global_words;
    if (!f->global_symbols.empty()) {
        global_words.reserve(f->global_symbols.size());
        launch_params.reserve(launch_params.size() + f->global_symbols.size());
        arg_info.reserve(arg_info.size() + f->global_symbols.size());
        if (arg_info.size() + f->global_symbols.size() > 31) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        for (const DriverGlobalSymbol& symbol : f->global_symbols) {
            void* storage = nullptr;
            const CUresult ready =
                ensure_module_global_storage(f->module, symbol, &storage);
            if (ready != CUDA_SUCCESS) return ready;
            global_words.push_back(reinterpret_cast<std::uintptr_t>(storage));
        }
        for (std::uintptr_t& word : global_words) {
            launch_params.push_back(&word);
            arg_info.push_back(cumetalKernelArgInfo_t{
                .kind = CUMETAL_ARG_BUFFER,
                .size_bytes = 0,
            });
        }
    }

    const cumetalKernel_t kernel{
        .metallib_path = function_module_path(f).c_str(),
        .kernel_name = f->kernel_name.c_str(),
        .arg_count = static_cast<std::uint32_t>(arg_info.size()),
        .arg_info = arg_info.empty() ? nullptr : arg_info.data(),
    };

    const cudaError_t status = cudaLaunchKernel(&kernel,
                                                dim3(gridDimX, gridDimY, gridDimZ),
                                                dim3(blockDimX, blockDimY, blockDimZ),
                                                launch_params.empty() ? nullptr : launch_params.data(),
                                                sharedMemBytes,
                                                reinterpret_cast<cudaStream_t>(hStream));
    return map_cuda_error(status);
}

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    if (dptr == nullptr || bytesize == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }

    void* allocated = nullptr;
    const cudaError_t status = cudaMalloc(&allocated, bytesize);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    *dptr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(allocated));
    return CUDA_SUCCESS;
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
    if (dptr == nullptr || bytesize == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }

    void* allocated = nullptr;
    const cudaError_t status = cudaMallocManaged(&allocated, bytesize, flags);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    *dptr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(allocated));
    return CUDA_SUCCESS;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    {
        DriverState& state = driver_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.vmm_reservations.contains(dptr)) {
             
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    return map_cuda_error(cudaFree(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dptr))));
}

CUresult cuMemGetAllocationGranularity(
    size_t* granularity, const CUmemAllocationProp* prop,
    CUmemAllocationGranularity_flags option) {
    if (granularity == nullptr || prop == nullptr ||
        (option != CU_MEM_ALLOC_GRANULARITY_MINIMUM &&
         option != CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    if (!valid_vmm_prop(*prop)) return CUDA_ERROR_INVALID_VALUE;
    *granularity = kDriverVmmGranularity;
    return CUDA_SUCCESS;
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags) {
    if (ptr == nullptr || size == 0 || size % kDriverVmmGranularity != 0 ||
        (alignment != 0 &&
         (!is_power_of_two(alignment) || alignment < kDriverVmmGranularity)) ||
        addr != 0 || flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;

    void* allocation = nullptr;
    const cudaError_t status = cudaMalloc(&allocation, size);
    if (status != cudaSuccess) return map_cuda_error(status);
    const CUdeviceptr device_ptr =
        static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(allocation));
    {
        DriverState& state = driver_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.vmm_reservations.emplace(
            device_ptr, DriverState::VmmReservationRecord{.size = size});
    }
    *ptr = device_ptr;
    return CUDA_SUCCESS;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    if (ptr == 0 || size == 0) return CUDA_ERROR_INVALID_VALUE;
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto found = state.vmm_reservations.find(ptr);
    if (found == state.vmm_reservations.end() || found->second.size != size ||
        found->second.mapped) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const cudaError_t status =
        cudaFree(reinterpret_cast<void*>(static_cast<std::uintptr_t>(ptr)));
    if (status != cudaSuccess) return map_cuda_error(status);
    state.vmm_reservations.erase(found);
    return CUDA_SUCCESS;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size,
                     const CUmemAllocationProp* prop,
                     unsigned long long flags) {
    if (handle == nullptr || prop == nullptr || size == 0 ||
        size % kDriverVmmGranularity != 0 || flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    if (!valid_vmm_prop(*prop)) return CUDA_ERROR_INVALID_VALUE;

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    CUmemGenericAllocationHandle candidate = state.next_vmm_handle++;
    if (candidate == 0) candidate = state.next_vmm_handle++;
    state.vmm_handles.emplace(
        candidate, DriverState::VmmHandleRecord{.size = size, .prop = *prop});
    *handle = candidate;
    return CUDA_SUCCESS;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    if (handle == 0) return CUDA_ERROR_INVALID_VALUE;
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.vmm_handles.erase(handle) == 1 ? CUDA_SUCCESS
                                                : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) {
    if (prop == nullptr || handle == 0) return CUDA_ERROR_INVALID_VALUE;
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto found = state.vmm_handles.find(handle);
    if (found == state.vmm_handles.end()) return CUDA_ERROR_INVALID_VALUE;
    *prop = found->second.prop;
    return CUDA_SUCCESS;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags) {
    if (ptr == 0 || size == 0 || offset != 0 || handle == 0 || flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto reservation = state.vmm_reservations.find(ptr);
    const auto allocation = state.vmm_handles.find(handle);
    if (reservation == state.vmm_reservations.end() ||
        allocation == state.vmm_handles.end() || reservation->second.mapped ||
        reservation->second.size != size || allocation->second.size != size) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    reservation->second.mapped = true;
    reservation->second.access_enabled = false;
    reservation->second.mapped_prop = allocation->second.prop;
    return CUDA_SUCCESS;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    if (ptr == 0 || size == 0) return CUDA_ERROR_INVALID_VALUE;
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto found = state.vmm_reservations.find(ptr);
    if (found == state.vmm_reservations.end() || found->second.size != size ||
        !found->second.mapped) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    found->second.mapped = false;
    found->second.access_enabled = false;
    return CUDA_SUCCESS;
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc* desc, size_t count) {
    if (ptr == 0 || size == 0 || desc == nullptr || count != 1 ||
        desc[0].location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
        desc[0].location.id != 0 ||
        (desc[0].flags != CU_MEM_ACCESS_FLAGS_PROT_READ &&
         desc[0].flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto found = state.vmm_reservations.find(ptr);
    if (found == state.vmm_reservations.end() || found->second.size != size ||
        !found->second.mapped) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    found->second.access_enabled = true;
    return CUDA_SUCCESS;
}

CUresult cuMemAllocHost(void** pp, size_t bytesize) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaHostAlloc(pp, bytesize, cudaHostAllocDefault));
}

CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int flags) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaHostAlloc(pp, bytesize, flags));
}

CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int flags) {
    if (pdptr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }

    void* device_ptr = nullptr;
    const cudaError_t status = cudaHostGetDevicePointer(&device_ptr, p, flags);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    *pdptr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(device_ptr));
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaHostGetFlags(pFlags, p));
}

CUresult cuMemFreeHost(void* p) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaFreeHost(p));
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                     srcHost,
                                     ByteCount,
                                     cudaMemcpyHostToDevice));
}

CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemcpy(dstHost,
                                     reinterpret_cast<void*>(static_cast<std::uintptr_t>(srcDevice)),
                                     ByteCount,
                                     cudaMemcpyDeviceToHost));
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                     reinterpret_cast<void*>(static_cast<std::uintptr_t>(srcDevice)),
                                     ByteCount,
                                     cudaMemcpyDeviceToDevice));
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice,
                           const void* srcHost,
                           size_t ByteCount,
                           CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemcpyAsync(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                          srcHost,
                                          ByteCount,
                                          cudaMemcpyHostToDevice,
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuMemcpyDtoHAsync(void* dstHost,
                           CUdeviceptr srcDevice,
                           size_t ByteCount,
                           CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemcpyAsync(dstHost,
                                          reinterpret_cast<void*>(static_cast<std::uintptr_t>(srcDevice)),
                                          ByteCount,
                                          cudaMemcpyDeviceToHost,
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice,
                           CUdeviceptr srcDevice,
                           size_t ByteCount,
                           CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemcpyAsync(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                          reinterpret_cast<void*>(static_cast<std::uintptr_t>(srcDevice)),
                                          ByteCount,
                                          cudaMemcpyDeviceToDevice,
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    return map_cuda_error(cudaMemcpyAsync(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)),
        reinterpret_cast<const void*>(static_cast<std::uintptr_t>(src)),
        byteCount, cudaMemcpyDefault,
        reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext  ,
                      CUdeviceptr srcDevice, CUcontext  ,
                      size_t ByteCount) {
    return cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext  ,
                            CUdeviceptr srcDevice, CUcontext  ,
                            size_t ByteCount, CUstream hStream) {
    return cuMemcpyDtoDAsync(dstDevice, srcDevice, ByteCount, hStream);
}

namespace {

struct ResolvedMemcpy3D {
    const char* src_base = nullptr;
    char* dst_base = nullptr;
    size_t src_pitch = 0;
    size_t dst_pitch = 0;
    size_t src_height = 0;
    size_t dst_height = 0;
    CUDA_MEMCPY3D copy{};
};

CUresult resolve_memcpy3d(const CUDA_MEMCPY3D* pCopy, ResolvedMemcpy3D* out) {
    if (pCopy == nullptr) return CUDA_ERROR_INVALID_VALUE;
    out->copy = *pCopy;
    out->src_pitch = pCopy->srcPitch ? pCopy->srcPitch : pCopy->WidthInBytes;
    out->dst_pitch = pCopy->dstPitch ? pCopy->dstPitch : pCopy->WidthInBytes;
    out->src_height = pCopy->srcHeight ? pCopy->srcHeight : pCopy->Height;
    out->dst_height = pCopy->dstHeight ? pCopy->dstHeight : pCopy->Height;
    if (pCopy->WidthInBytes == 0 || pCopy->Height == 0 || pCopy->Depth == 0) return CUDA_SUCCESS;
    const size_t src_span =
        (pCopy->srcZ + pCopy->Depth - 1) * out->src_pitch * out->src_height +
        (pCopy->srcY + pCopy->Height - 1) * out->src_pitch + pCopy->srcXInBytes + pCopy->WidthInBytes;
    const size_t dst_span =
        (pCopy->dstZ + pCopy->Depth - 1) * out->dst_pitch * out->dst_height +
        (pCopy->dstY + pCopy->Height - 1) * out->dst_pitch + pCopy->dstXInBytes + pCopy->WidthInBytes;
    out->src_base =
        (pCopy->srcMemoryType == CU_MEMORYTYPE_HOST)
            ? static_cast<const char*>(pCopy->srcHost)
            : static_cast<const char*>(cumetalRuntimeGetHostPointer(
                  reinterpret_cast<const void*>(static_cast<std::uintptr_t>(pCopy->srcDevice)),
                  src_span));
    out->dst_base =
        (pCopy->dstMemoryType == CU_MEMORYTYPE_HOST)
            ? static_cast<char*>(pCopy->dstHost)
            : static_cast<char*>(cumetalRuntimeGetHostPointer(
                  reinterpret_cast<const void*>(static_cast<std::uintptr_t>(pCopy->dstDevice)),
                  dst_span));
    if (out->src_base == nullptr || out->dst_base == nullptr) return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}

void perform_memcpy3d(const ResolvedMemcpy3D& resolved) {
    const CUDA_MEMCPY3D& copy = resolved.copy;
    if (copy.WidthInBytes == 0 || copy.Height == 0 || copy.Depth == 0) return;
    for (size_t z = 0; z < copy.Depth; ++z) {
        const size_t sz_off = (copy.srcZ + z) * resolved.src_pitch * resolved.src_height;
        const size_t dz_off = (copy.dstZ + z) * resolved.dst_pitch * resolved.dst_height;
        for (size_t y = 0; y < copy.Height; ++y) {
            const char* src_row =
                resolved.src_base + sz_off + (copy.srcY + y) * resolved.src_pitch + copy.srcXInBytes;
            char* dst_row =
                resolved.dst_base + dz_off + (copy.dstY + y) * resolved.dst_pitch + copy.dstXInBytes;
            std::memcpy(dst_row, src_row, copy.WidthInBytes);
        }
    }
}

}   

CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy) {
    if (pCopy == nullptr) return CUDA_ERROR_INVALID_VALUE;
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    ResolvedMemcpy3D resolved;
    const CUresult status = resolve_memcpy3d(pCopy, &resolved);
    if (status != CUDA_SUCCESS) return status;
    perform_memcpy3d(resolved);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream) {
    if (pCopy == nullptr) return CUDA_ERROR_INVALID_VALUE;
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    if (pCopy->WidthInBytes == 0 || pCopy->Height == 0 || pCopy->Depth == 0) {
        return CUDA_SUCCESS;
    }
    if ((pCopy->srcMemoryType == CU_MEMORYTYPE_HOST && pCopy->srcHost == nullptr) ||
        (pCopy->srcMemoryType == CU_MEMORYTYPE_DEVICE && pCopy->srcDevice == 0) ||
        (pCopy->dstMemoryType == CU_MEMORYTYPE_HOST && pCopy->dstHost == nullptr) ||
        (pCopy->dstMemoryType == CU_MEMORYTYPE_DEVICE && pCopy->dstDevice == 0)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto* payload = new (std::nothrow) ResolvedMemcpy3D;
    if (payload == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    const CUresult status = resolve_memcpy3d(pCopy, payload);
    if (status != CUDA_SUCCESS) {
        delete payload;
        return status;
    }
    const cudaError_t launched = cudaLaunchHostFunc(
        reinterpret_cast<cudaStream_t>(hStream),
        +[](void* raw) {
            std::unique_ptr<ResolvedMemcpy3D> owned(static_cast<ResolvedMemcpy3D*>(raw));
            perform_memcpy3d(*owned);
        },
        payload);
    if (launched != cudaSuccess) delete payload;
    return map_cuda_error(launched);
}

namespace {

const char* memcpy2d_source(const CUDA_MEMCPY2D& copy, size_t span) {
    switch (copy.srcMemoryType) {
        case CU_MEMORYTYPE_HOST:
            return static_cast<const char*>(copy.srcHost);
        case CU_MEMORYTYPE_DEVICE:
        case CU_MEMORYTYPE_UNIFIED:
            return static_cast<const char*>(cumetalRuntimeGetHostPointer(
                reinterpret_cast<const void*>(static_cast<std::uintptr_t>(copy.srcDevice)), span));
        default:
            return nullptr;
    }
}

char* memcpy2d_destination(const CUDA_MEMCPY2D& copy, size_t span) {
    switch (copy.dstMemoryType) {
        case CU_MEMORYTYPE_HOST:
            return static_cast<char*>(copy.dstHost);
        case CU_MEMORYTYPE_DEVICE:
        case CU_MEMORYTYPE_UNIFIED:
            return static_cast<char*>(cumetalRuntimeGetHostPointer(
                reinterpret_cast<const void*>(static_cast<std::uintptr_t>(copy.dstDevice)), span));
        default:
            return nullptr;
    }
}

CUresult validate_memcpy2d(const CUDA_MEMCPY2D* pCopy) {
    if (pCopy == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (pCopy->srcMemoryType == CU_MEMORYTYPE_ARRAY ||
        pCopy->dstMemoryType == CU_MEMORYTYPE_ARRAY) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    const size_t src_pitch = pCopy->srcPitch ? pCopy->srcPitch : pCopy->WidthInBytes;
    const size_t dst_pitch = pCopy->dstPitch ? pCopy->dstPitch : pCopy->WidthInBytes;
    if (pCopy->WidthInBytes > src_pitch || pCopy->WidthInBytes > dst_pitch) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

}   

namespace {

struct ResolvedMemcpy2D {
    const char* src_base = nullptr;
    char* dst_base = nullptr;
    size_t src_pitch = 0;
    size_t dst_pitch = 0;
    CUDA_MEMCPY2D copy{};
};

CUresult resolve_memcpy2d(const CUDA_MEMCPY2D* pCopy, ResolvedMemcpy2D* out) {
    const CUresult valid = validate_memcpy2d(pCopy);
    if (valid != CUDA_SUCCESS) return valid;
    out->copy = *pCopy;
    out->src_pitch = pCopy->srcPitch ? pCopy->srcPitch : pCopy->WidthInBytes;
    out->dst_pitch = pCopy->dstPitch ? pCopy->dstPitch : pCopy->WidthInBytes;
    if (pCopy->WidthInBytes == 0 || pCopy->Height == 0) return CUDA_SUCCESS;
    const size_t src_span =
        (pCopy->srcY + pCopy->Height - 1) * out->src_pitch + pCopy->srcXInBytes + pCopy->WidthInBytes;
    const size_t dst_span =
        (pCopy->dstY + pCopy->Height - 1) * out->dst_pitch + pCopy->dstXInBytes + pCopy->WidthInBytes;
    out->src_base = memcpy2d_source(*pCopy, src_span);
    out->dst_base = memcpy2d_destination(*pCopy, dst_span);
    if (out->src_base == nullptr || out->dst_base == nullptr) return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}

bool memcpy2d_touches_pageable_host(const CUDA_MEMCPY2D& copy) {
    const bool src_pageable = copy.srcMemoryType == CU_MEMORYTYPE_HOST &&
                              copy.srcHost != nullptr &&
                              cumetalRuntimeIsDevicePointer(copy.srcHost) == 0;
    const bool dst_pageable = copy.dstMemoryType == CU_MEMORYTYPE_HOST &&
                              copy.dstHost != nullptr &&
                              cumetalRuntimeIsDevicePointer(copy.dstHost) == 0;
    return src_pageable || dst_pageable;
}

void perform_memcpy2d(const ResolvedMemcpy2D& resolved) {
    const CUDA_MEMCPY2D& copy = resolved.copy;
    if (copy.WidthInBytes == 0 || copy.Height == 0) return;
    for (size_t y = 0; y < copy.Height; ++y) {
        const char* src_row =
            resolved.src_base + (copy.srcY + y) * resolved.src_pitch + copy.srcXInBytes;
        char* dst_row = resolved.dst_base + (copy.dstY + y) * resolved.dst_pitch + copy.dstXInBytes;
        std::memmove(dst_row, src_row, copy.WidthInBytes);
    }
}

}   

CUresult cuMemcpy2D(const CUDA_MEMCPY2D* pCopy) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    ResolvedMemcpy2D resolved;
    const CUresult status = resolve_memcpy2d(pCopy, &resolved);
    if (status != CUDA_SUCCESS) return status;
    perform_memcpy2d(resolved);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    auto* payload = new (std::nothrow) ResolvedMemcpy2D;
    if (payload == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    const CUresult status = resolve_memcpy2d(pCopy, payload);
    if (status != CUDA_SUCCESS) {
        delete payload;
        return status;
    }
    if (payload->copy.WidthInBytes == 0 || payload->copy.Height == 0) {
        delete payload;
        return CUDA_SUCCESS;
    }

    const bool pageable = memcpy2d_touches_pageable_host(payload->copy);
    const cudaError_t launched = cudaLaunchHostFunc(
        reinterpret_cast<cudaStream_t>(hStream),
        +[](void* raw) {
            std::unique_ptr<ResolvedMemcpy2D> owned(static_cast<ResolvedMemcpy2D*>(raw));
            perform_memcpy2d(*owned);
        },
        payload);
    if (launched != cudaSuccess) {
        delete payload;
        return map_cuda_error(launched);
    }
    if (pageable) {
        return cuStreamSynchronize(hStream);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemcpyBatchAsync(CUdeviceptr* dsts, CUdeviceptr* srcs, size_t* sizes, size_t count,
                            CUmemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs,
                            size_t* failIdx, CUstream hStream) {
    if (failIdx != nullptr) *failIdx = SIZE_MAX;
    if (count == 0) return CUDA_SUCCESS;
    if (dsts == nullptr || srcs == nullptr || sizes == nullptr) return CUDA_ERROR_INVALID_VALUE;

    if (numAttrs != 0) {
        if (attrs == nullptr || attrsIdxs == nullptr || attrsIdxs[0] != 0) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        for (size_t i = 1; i < numAttrs; ++i) {
            if (attrsIdxs[i] <= attrsIdxs[i - 1] || attrsIdxs[i] >= count) {
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
        for (size_t i = 0; i < numAttrs; ++i) {
            if (attrs[i].srcAccessOrder == CU_MEMCPY_SRC_ACCESS_ORDER_INVALID) {
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    for (size_t i = 0; i < count; ++i) {
        const cudaError_t status = cudaMemcpyAsync(
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(dsts[i])),
            reinterpret_cast<const void*>(static_cast<std::uintptr_t>(srcs[i])),
            sizes[i], cudaMemcpyDefault, reinterpret_cast<cudaStream_t>(hStream));
        if (status != cudaSuccess) {
            if (failIdx != nullptr) *failIdx = i;
            return map_cuda_error(status);
        }
    }
    return CUDA_SUCCESS;
}

CUresult cuMemGetInfo(size_t* freeBytes, size_t* totalBytes) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemGetInfo(freeBytes, totalBytes));
}

CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                     static_cast<int>(uc),
                                     N));
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemsetAsync(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                          static_cast<int>(uc),
                                          N,
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuGetErrorName(CUresult error, const char** pStr) {
    if (pStr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    switch (error) {
        case CUDA_SUCCESS:
            *pStr = "CUDA_SUCCESS";
            break;
        case CUDA_ERROR_INVALID_VALUE:
            *pStr = "CUDA_ERROR_INVALID_VALUE";
            break;
        case CUDA_ERROR_OUT_OF_MEMORY:
            *pStr = "CUDA_ERROR_OUT_OF_MEMORY";
            break;
        case CUDA_ERROR_NOT_INITIALIZED:
            *pStr = "CUDA_ERROR_NOT_INITIALIZED";
            break;
        case CUDA_ERROR_DEVICES_UNAVAILABLE:
            *pStr = "CUDA_ERROR_DEVICES_UNAVAILABLE";
            break;
        case CUDA_ERROR_INVALID_DEVICE:
            *pStr = "CUDA_ERROR_INVALID_DEVICE";
            break;
        case CUDA_ERROR_INVALID_IMAGE:
            *pStr = "CUDA_ERROR_INVALID_IMAGE";
            break;
        case CUDA_ERROR_INVALID_CONTEXT:
            *pStr = "CUDA_ERROR_INVALID_CONTEXT";
            break;
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            *pStr = "CUDA_ERROR_NO_BINARY_FOR_GPU";
            break;
        case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
            *pStr = "CUDA_ERROR_JIT_COMPILER_NOT_FOUND";
            break;
        case CUDA_ERROR_FILE_NOT_FOUND:
            *pStr = "CUDA_ERROR_FILE_NOT_FOUND";
            break;
        case CUDA_ERROR_NOT_FOUND:
            *pStr = "CUDA_ERROR_NOT_FOUND";
            break;
        case CUDA_ERROR_NOT_READY:
            *pStr = "CUDA_ERROR_NOT_READY";
            break;
        case CUDA_ERROR_ILLEGAL_ADDRESS:
            *pStr = "CUDA_ERROR_ILLEGAL_ADDRESS";
            break;
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            *pStr = "CUDA_ERROR_LAUNCH_TIMEOUT";
            break;
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            *pStr = "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
            break;
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            *pStr = "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
            break;
        case CUDA_ERROR_DEINITIALIZED:
            *pStr = "CUDA_ERROR_DEINITIALIZED";
            break;
        case CUDA_ERROR_NO_DEVICE:
            *pStr = "CUDA_ERROR_NO_DEVICE";
            break;
        case CUDA_ERROR_INVALID_HANDLE:
            *pStr = "CUDA_ERROR_INVALID_HANDLE";
            break;
        case CUDA_ERROR_LAUNCH_FAILED:
            *pStr = "CUDA_ERROR_LAUNCH_FAILED";
            break;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            *pStr = "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
            break;
        case CUDA_ERROR_ASSERT:
            *pStr = "CUDA_ERROR_ASSERT";
            break;
        case CUDA_ERROR_NOT_PERMITTED:
            *pStr = "CUDA_ERROR_NOT_PERMITTED";
            break;
        case CUDA_ERROR_NOT_SUPPORTED:
            *pStr = "CUDA_ERROR_NOT_SUPPORTED";
            break;
        case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
            *pStr = "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE";
            break;
        case CUDA_ERROR_UNKNOWN:
            *pStr = "CUDA_ERROR_UNKNOWN";
            break;
        default:
            *pStr = "CUDA_ERROR_UNKNOWN";
            break;
    }
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char** pStr) {
    return cuGetErrorName(error, pStr);
}

CUresult cuProfilerStart(void) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return CUDA_SUCCESS;
}

CUresult cuProfilerStop(void) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return CUDA_SUCCESS;
}

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int  ) {
    return cuStreamCreate(phStream, flags);
}

CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = 0;
    return CUDA_SUCCESS;
}

CUresult cuLaunchCooperativeKernel(CUfunction f,
                                    unsigned int gridDimX, unsigned int gridDimY,
                                    unsigned int gridDimZ, unsigned int blockDimX,
                                    unsigned int blockDimY, unsigned int blockDimZ,
                                    unsigned int sharedMemBytes, CUstream hStream,
                                    void** kernelParams) {
    const std::uint64_t block_count =
        static_cast<std::uint64_t>(gridDimX) * gridDimY * gridDimZ;
    if (block_count == 0 || block_count > 4) {
        return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    }
    return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                          sharedMemBytes, hStream, kernelParams, nullptr);
}

CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    if (dstDevice == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto* ptr = static_cast<unsigned short*>(cumetalRuntimeGetHostPointer(
        reinterpret_cast<const void*>(static_cast<uintptr_t>(dstDevice)),
        N * sizeof(unsigned short)));
    if (ptr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (size_t i = 0; i < N; ++i) {
        ptr[i] = us;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    if (dstDevice == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto* ptr = static_cast<unsigned int*>(cumetalRuntimeGetHostPointer(
        reinterpret_cast<const void*>(static_cast<uintptr_t>(dstDevice)),
        N * sizeof(unsigned int)));
    if (ptr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (size_t i = 0; i < N; ++i) {
        ptr[i] = ui;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N,
                           CUstream  ) {
    return cuMemsetD16(dstDevice, us, N);
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N,
                           CUstream  ) {
    return cuMemsetD32(dstDevice, ui, N);
}

CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch,
                       unsigned char uc, size_t Width, size_t Height) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    auto* base = static_cast<unsigned char*>(cumetalRuntimeGetHostPointer(
        reinterpret_cast<const void*>(static_cast<std::uintptr_t>(dstDevice)),
        Height == 0 ? 0 : (Height - 1) * dstPitch + Width));
    if (base == nullptr && Width > 0 && Height > 0) return CUDA_ERROR_INVALID_VALUE;
    for (size_t row = 0; row < Height; ++row) {
        std::memset(base + row * dstPitch, static_cast<int>(uc), Width);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch,
                        unsigned short us, size_t Width, size_t Height) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    auto* base = static_cast<unsigned char*>(cumetalRuntimeGetHostPointer(
        reinterpret_cast<const void*>(static_cast<std::uintptr_t>(dstDevice)),
        Height == 0 ? 0 : (Height - 1) * dstPitch + Width * sizeof(unsigned short)));
    if (base == nullptr && Width > 0 && Height > 0) return CUDA_ERROR_INVALID_VALUE;
    for (size_t row = 0; row < Height; ++row) {
        auto* row_ptr = reinterpret_cast<unsigned short*>(base + row * dstPitch);
        for (size_t col = 0; col < Width; ++col) row_ptr[col] = us;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch,
                        unsigned int ui, size_t Width, size_t Height) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    auto* base = static_cast<unsigned char*>(cumetalRuntimeGetHostPointer(
        reinterpret_cast<const void*>(static_cast<std::uintptr_t>(dstDevice)),
        Height == 0 ? 0 : (Height - 1) * dstPitch + Width * sizeof(unsigned int)));
    if (base == nullptr && Width > 0 && Height > 0) return CUDA_ERROR_INVALID_VALUE;
    for (size_t row = 0; row < Height; ++row) {
        auto* row_ptr = reinterpret_cast<unsigned int*>(base + row * dstPitch);
        for (size_t col = 0; col < Width; ++col) row_ptr[col] = ui;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8Async(CUdeviceptr d, size_t p, unsigned char uc,
                            size_t W, size_t H, CUstream s) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    auto* base = static_cast<unsigned char*>(cumetalRuntimeGetHostPointer(
        reinterpret_cast<const void*>(static_cast<std::uintptr_t>(d)),
        H == 0 ? 0 : (H - 1) * p + W));
    if (base == nullptr && W > 0 && H > 0) return CUDA_ERROR_INVALID_VALUE;
    struct Payload { unsigned char* base; size_t pitch, width, height; unsigned char value; };
    auto* payload = new (std::nothrow) Payload{base, p, W, H, uc};
    if (!payload) return CUDA_ERROR_OUT_OF_MEMORY;
    const cudaError_t status = cudaLaunchHostFunc(
        reinterpret_cast<cudaStream_t>(s),
        [](void* opaque) {
            std::unique_ptr<Payload> data(static_cast<Payload*>(opaque));
            for (size_t row = 0; row < data->height; ++row)
                std::memset(data->base + row * data->pitch, data->value, data->width);
        }, payload);
    if (status != cudaSuccess) delete payload;
    return map_cuda_error(status);
}
CUresult cuMemsetD2D16Async(CUdeviceptr d, size_t p, unsigned short us,
                             size_t W, size_t H, CUstream s) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    auto* base = static_cast<unsigned char*>(cumetalRuntimeGetHostPointer(
        reinterpret_cast<const void*>(static_cast<std::uintptr_t>(d)),
        H == 0 ? 0 : (H - 1) * p + W * sizeof(unsigned short)));
    if (base == nullptr && W > 0 && H > 0) return CUDA_ERROR_INVALID_VALUE;
    struct Payload { unsigned char* base; size_t pitch, width, height; unsigned short value; };
    auto* payload = new (std::nothrow) Payload{base, p, W, H, us};
    if (!payload) return CUDA_ERROR_OUT_OF_MEMORY;
    const cudaError_t status = cudaLaunchHostFunc(
        reinterpret_cast<cudaStream_t>(s),
        [](void* opaque) {
            std::unique_ptr<Payload> data(static_cast<Payload*>(opaque));
            for (size_t row = 0; row < data->height; ++row) {
                auto* values = reinterpret_cast<unsigned short*>(
                    data->base + row * data->pitch);
                std::fill_n(values, data->width, data->value);
            }
        }, payload);
    if (status != cudaSuccess) delete payload;
    return map_cuda_error(status);
}
CUresult cuMemsetD2D32Async(CUdeviceptr d, size_t p, unsigned int ui,
                             size_t W, size_t H, CUstream s) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    auto* base = static_cast<unsigned char*>(cumetalRuntimeGetHostPointer(
        reinterpret_cast<const void*>(static_cast<std::uintptr_t>(d)),
        H == 0 ? 0 : (H - 1) * p + W * sizeof(unsigned int)));
    if (base == nullptr && W > 0 && H > 0) return CUDA_ERROR_INVALID_VALUE;
    struct Payload { unsigned char* base; size_t pitch, width, height; unsigned int value; };
    auto* payload = new (std::nothrow) Payload{base, p, W, H, ui};
    if (!payload) return CUDA_ERROR_OUT_OF_MEMORY;
    const cudaError_t status = cudaLaunchHostFunc(
        reinterpret_cast<cudaStream_t>(s),
        [](void* opaque) {
            std::unique_ptr<Payload> data(static_cast<Payload*>(opaque));
            for (size_t row = 0; row < data->height; ++row) {
                auto* values = reinterpret_cast<unsigned int*>(
                    data->base + row * data->pitch);
                std::fill_n(values, data->width, data->value);
            }
        }, payload);
    if (status != cudaSuccess) delete payload;
    return map_cuda_error(status);
}

CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
    if (pbase == nullptr && psize == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    void* base = nullptr;
    size_t sz = 0;
    const void* raw = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(dptr));
    if (!cumetalRuntimeGetAllocationInfo(raw, &base, &sz)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (pbase) *pbase = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(base));
    if (psize) *psize = sz;
    return CUDA_SUCCESS;
}

CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    if (data == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    const void* raw = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(ptr));
    const bool is_known = (cumetalRuntimeIsDevicePointer(raw) != 0);
    switch (attribute) {
        case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: {
            auto* out = static_cast<unsigned int*>(data);

            *out = is_known ? static_cast<unsigned int>(CU_MEMORYTYPE_UNIFIED)
                            : static_cast<unsigned int>(CU_MEMORYTYPE_HOST);
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_DEVICE_POINTER: {
            auto* out = static_cast<CUdeviceptr*>(data);
            *out = ptr;   
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_HOST_POINTER: {
            auto* out = static_cast<void**>(data);
            *out = reinterpret_cast<void*>(static_cast<std::uintptr_t>(ptr));
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_IS_MANAGED: {
            auto* out = static_cast<unsigned int*>(data);
            *out = static_cast<unsigned int>(cumetalRuntimeIsManaged(raw));
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_MAPPED: {
            auto* out = static_cast<unsigned int*>(data);
            *out = is_known ? 1u : 0u;
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE: {

            auto* out = static_cast<void**>(data);
            *out = nullptr;
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_CONTEXT: {
            auto* out = static_cast<CUcontext*>(data);
            cuCtxGetCurrent(out);
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: {
             
            if (!is_known) return CUDA_ERROR_INVALID_VALUE;
            *static_cast<int*>(data) = 0;
            return CUDA_SUCCESS;
        }
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult cuPointerGetAttributes(unsigned int numAttributes,
                                CUpointer_attribute* attributes,
                                void** data,
                                CUdeviceptr ptr) {
    if (numAttributes == 0) return CUDA_SUCCESS;
    if (attributes == nullptr || data == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    for (unsigned int i = 0; i < numAttributes; ++i) {
        if (data[i] == nullptr) continue;
         
        (void) cuPointerGetAttribute(data[i], attributes[i], ptr);
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) {
    if (major == nullptr || minor == nullptr || dev != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *major = 8;
    *minor = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice  , CUdevice  ) {
    if (canAccessPeer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *canAccessPeer = 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxEnablePeerAccess(CUcontext  , unsigned int  ) {
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuCtxDisablePeerAccess(CUcontext  ) {
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes,
                          size_t Height, unsigned int  ) {
    if (dptr == nullptr || pPitch == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    constexpr size_t kAlign = 512;
    *pPitch = (WidthInBytes + kAlign - 1) & ~(kAlign - 1);
    return cuMemAlloc(dptr, *pPitch * Height);
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                     CUfunction func,
                                                     int blockSize,
                                                     size_t dynamicSMemSize) {
    if (numBlocks == nullptr || blockSize <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    cumetal::metal_backend::KernelProperties kernel{};
    const CUresult query = query_function_properties(func, &kernel);
    if (query != CUDA_SUCCESS) return query;
    if (blockSize > kernel.max_threads_per_threadgroup)
        return CUDA_ERROR_INVALID_VALUE;
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess)
        return CUDA_ERROR_INVALID_VALUE;
    const int thread_bound =
        std::max(1, kernel.max_threads_per_threadgroup / blockSize);
    int memory_bound = thread_bound;
    const size_t total_shared =
        kernel.static_threadgroup_memory_bytes + dynamicSMemSize;
    if (total_shared > static_cast<size_t>(device.sharedMemPerBlock))
        return CUDA_ERROR_INVALID_VALUE;
    if (total_shared > 0)
        memory_bound = std::max(
            1, static_cast<int>(device.sharedMemPerBlock / total_shared));
    *numBlocks = std::min(thread_bound, memory_bound);
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize,
                                          int* blockSize,
                                          CUfunction func,
                                          CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                          size_t dynamicSMemSize,
                                          int blockSizeLimit) {
    if (minGridSize == nullptr || blockSize == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    cumetal::metal_backend::KernelProperties kernel{};
    const CUresult query = query_function_properties(func, &kernel);
    if (query != CUDA_SUCCESS) return query;
    int chosen = kernel.max_threads_per_threadgroup;
    if (blockSizeLimit > 0) chosen = std::min(chosen, blockSizeLimit);
    const int width = std::max(1, kernel.thread_execution_width);
    chosen = std::max(width, (chosen / width) * width);
     
    if (blockSizeToDynamicSMemSize != nullptr) {
        dynamicSMemSize = blockSizeToDynamicSMemSize(chosen);
    }
    int active_blocks = 0;
    const CUresult occupancy = cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, func, chosen, dynamicSMemSize);
    if (occupancy != CUDA_SUCCESS) return occupancy;
    *blockSize = chosen;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.multiProcessorCount > 0) {
        *minGridSize = prop.multiProcessorCount * active_blocks;
    } else {
        *minGridSize = 16;
    }
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize,
                                                   int* blockSize,
                                                   CUfunction func,
                                                   CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                                   size_t dynamicSMemSize,
                                                   int blockSizeLimit,
                                                   unsigned int flags) {
    if ((flags & ~static_cast<unsigned int>(CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE)) != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func,
                                            blockSizeToDynamicSMemSize, dynamicSMemSize,
                                            blockSizeLimit);
}

CUresult cuFuncLoad(CUfunction function) {
    std::string metallib_path;
    std::string kernel_name;
    {
        DriverState& state = driver_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) return CUDA_ERROR_NOT_INITIALIZED;
        if (!has_current_context_locked(state)) return CUDA_ERROR_INVALID_CONTEXT;
        if (!is_valid_function_locked(state, function) || function->module == nullptr)
            return CUDA_ERROR_INVALID_HANDLE;
        metallib_path = function_module_path(function);
        kernel_name = function->kernel_name;
    }

    std::string error;
    const cudaError_t prepared = cumetal::metal_backend::prepare_kernel_pipeline(
        metallib_path, kernel_name, &error);
    if (prepared != cudaSuccess && !error.empty()) {

        cumetal::warn_once("cuFuncLoad:" + metallib_path + "::" + kernel_name,
                           "cuFuncLoad failed to prepare '" + kernel_name + "': " + error);
    }
    return map_cuda_error(prepared);
}

CUresult cuFuncIsLoaded(CUfunctionLoadingState* state_out, CUfunction function) {
    if (state_out == nullptr) return CUDA_ERROR_INVALID_VALUE;
    std::string metallib_path;
    std::string kernel_name;
    {
        DriverState& state = driver_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) return CUDA_ERROR_NOT_INITIALIZED;
        if (!has_current_context_locked(state)) return CUDA_ERROR_INVALID_CONTEXT;

        if (!is_valid_function_locked(state, function) || function->module == nullptr)
            return CUDA_ERROR_INVALID_HANDLE;
        metallib_path = function_module_path(function);
        kernel_name = function->kernel_name;
    }

    *state_out = cumetal::metal_backend::kernel_pipeline_is_ready(metallib_path, kernel_name)
                     ? CU_FUNCTION_LOADING_STATE_LOADED
                     : CU_FUNCTION_LOADING_STATE_UNLOADED;
    return CUDA_SUCCESS;
}

CUresult cuFuncGetAttribute(int* pi, CUfunc_attribute attrib, CUfunction hfunc) {
    if (pi == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    cumetal::metal_backend::KernelProperties kernel{};
    const CUresult query = query_function_properties(hfunc, &kernel);
    if (query != CUDA_SUCCESS) return query;
    switch (attrib) {
        case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            *pi = kernel.max_threads_per_threadgroup;
            break;
        case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
            *pi = static_cast<int>(kernel.static_threadgroup_memory_bytes);
            break;
        case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: {
            cudaDeviceProp device{};
            if (cudaGetDeviceProperties(&device, 0) != cudaSuccess)
                return CUDA_ERROR_INVALID_VALUE;
            *pi = kernel.static_threadgroup_memory_bytes >=
                          static_cast<size_t>(device.sharedMemPerBlock)
                      ? 0
                      : static_cast<int>(
                            static_cast<size_t>(device.sharedMemPerBlock) -
                            kernel.static_threadgroup_memory_bytes);
            break;
        }
        case CU_FUNC_ATTRIBUTE_NUM_REGS:
        case CU_FUNC_ATTRIBUTE_PTX_VERSION:
        case CU_FUNC_ATTRIBUTE_BINARY_VERSION: {
             
            cudaFuncAttributes attributes{};
            const cudaError_t status = cumetalFuncAttributesFromKernelProperties(kernel, &attributes);
            if (status != cudaSuccess) return CUDA_ERROR_INVALID_VALUE;
            *pi = attrib == CU_FUNC_ATTRIBUTE_NUM_REGS      ? attributes.numRegs
                  : attrib == CU_FUNC_ATTRIBUTE_PTX_VERSION ? attributes.ptxVersion
                                                            : attributes.binaryVersion;
            break;
        }
        case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
        case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
        case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA:
        case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:

            *pi = 0;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
    if (config < CU_FUNC_CACHE_PREFER_NONE || config > CU_FUNC_CACHE_PREFER_EQUAL)
        return CUDA_ERROR_INVALID_VALUE;
    cumetal::metal_backend::KernelProperties kernel{};
    const CUresult query = query_function_properties(hfunc, &kernel);
    if (query != CUDA_SUCCESS) return query;
    return CUDA_SUCCESS;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunc_attribute attrib, int value) {
    if (attrib != CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES &&
        attrib != CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    cumetal::metal_backend::KernelProperties kernel{};
    const CUresult query = query_function_properties(hfunc, &kernel);
    if (query != CUDA_SUCCESS) return query;
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess)
        return CUDA_ERROR_INVALID_VALUE;
    const size_t max_dynamic = kernel.static_threadgroup_memory_bytes >=
                                       static_cast<size_t>(device.sharedMemPerBlock)
                                   ? 0
                                   : static_cast<size_t>(device.sharedMemPerBlock) -
                                         kernel.static_threadgroup_memory_bytes;
    if ((attrib == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES &&
         (value < 0 || static_cast<size_t>(value) > max_dynamic)) ||
        (attrib == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT &&
         (value < -1 || value > 100))) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                              CUfunction func,
                                                              int blockSize,
                                                              size_t dynamicSMemSize,
                                                              unsigned int flags) {
    if (flags != 0) return CUDA_ERROR_INVALID_VALUE;
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags) {
    if (phGraph == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    cudaGraph_t g = nullptr;
    cudaError_t err = cudaGraphCreate(&g, flags);
    *phGraph = reinterpret_cast<CUgraph>(g);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuGraphDestroy(CUgraph hGraph) {
    cudaError_t err = cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(hGraph));
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuGraphInstantiate(CUgraphExec* phGraphExec, CUgraph hGraph,
                             CUgraphNode* phErrorNode, char* logBuffer,
                             size_t bufferSize) {
    if (phGraphExec == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t errNode = nullptr;
    cudaError_t err = cudaGraphInstantiate(
        &exec, reinterpret_cast<cudaGraph_t>(hGraph), &errNode, logBuffer, bufferSize);
    *phGraphExec = reinterpret_cast<CUgraphExec>(exec);
    if (phErrorNode) { *phErrorNode = reinterpret_cast<CUgraphNode>(errNode); }
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    cudaError_t err = cudaGraphLaunch(
        reinterpret_cast<cudaGraphExec_t>(hGraphExec),
        reinterpret_cast<cudaStream_t>(hStream));
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
    cudaError_t err = cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(hGraphExec));
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) {
    if (pHandle == nullptr || pAllocateArray == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pHandle = nullptr;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuArrayDestroy(CUarray hArray) {
    return hArray == nullptr ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuTexObjectCreate(CUtexObject* pTexObject,
                           const CUDA_RESOURCE_DESC* pResDesc,
                           const CUDA_TEXTURE_DESC* pTexDesc,
                           const CUDA_RESOURCE_VIEW_DESC*  ) {
    if (pTexObject == nullptr || pResDesc == nullptr || pTexDesc == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pTexObject = 0;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuTexObjectDestroy(CUtexObject texObject) {
    return texObject == 0 ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
}

}

#include "cumetal/ptx/kernel_abi.h"
#include "cumetal/ptx/lower_to_llvm.h"

#include <algorithm>
#include <sstream>

namespace cumetal::ptx {

std::string build_ir_abi_sidecar(const cumetal::ir::Module& module,
                                 const std::string& requested_entry,
                                 std::string_view ptx_source, std::string* diagnostics) {
    std::ostringstream messages;
    std::string text = "CUMETAL_ABI_V2\n";
    std::size_t emitted = 0;
    for (const auto& function : module.functions) {
        if (!function.is_kernel || !function.kernel_abi.has_value()) {
            continue;
        }
        if (!requested_entry.empty() && function.name != requested_entry) {
            continue;
        }
        const cumetal::ir::KernelAbi& abi = *function.kernel_abi;
        std::string block = "kernel " + function.name + "\n";

        const auto shared = ptx_source.empty() ? abi.static_threadgroup_memory
            : cumetal::ptx::compute_static_shared_bytes(ptx_source, function.name);
        block += "shared " + std::to_string(shared) + "\n";
        bool usable = true;

        std::string globals;
        for (const auto& argument : abi.arguments) {
            constexpr std::string_view kGlobalRole = "global_symbol:";
            if (!argument.hidden_role.has_value()) continue;
            if (!argument.hidden_role->starts_with(kGlobalRole)) continue;
            const std::string name = argument.hidden_role->substr(kGlobalRole.size());
            const auto binding = std::find_if(
                abi.bindings.begin(), abi.bindings.end(),
                [&](const cumetal::ir::BindingDescriptor& candidate) {
                    return candidate.hidden_role == argument.hidden_role;
                });
            if (binding == abi.bindings.end() || binding->size == 0) {
                messages << "cumetalc: kernel ABI omitted for '" << function.name
                          << "': device global '" << name << "' has no sized binding\n";
                usable = false;
                break;
            }
            std::vector<std::uint8_t> initializer;
            for (const auto& symbol : module.external_symbols) {
                if (symbol.name == name) initializer = symbol.initializer;
            }
            if (initializer.empty() && !ptx_source.empty()) {
                if (const auto bytes =
                        cumetal::ptx::find_initialized_global_bytes(ptx_source, name))
                    initializer = *bytes;
            }
            globals += "global " + name + " " + std::to_string(binding->size) + " " +
                       std::to_string(std::max<std::uint32_t>(1, binding->alignment)) + " ";
            if (initializer.size() == binding->size) {
                static constexpr char kHex[] = "0123456789abcdef";
                for (const std::uint8_t byte : initializer) {
                    globals += kHex[byte >> 4];
                    globals += kHex[byte & 0xF];
                }
            } else {
                globals += "-";
            }
            globals += "\n";
        }
        if (!usable) continue;
        for (const auto& argument : abi.arguments) {
            if (argument.hidden_role.has_value()) continue;
            if (argument.kind == cumetal::ir::ArgumentKind::kPointer) {
                block += "arg buffer 8\n";
                continue;
            }
            if (argument.size == 0 || argument.size > 4096) {

                messages << "cumetalc: kernel ABI omitted for '" << function.name
                          << "': argument '" << argument.name << "' has unusable size "
                          << argument.size << "\n";
                usable = false;
                break;
            }
            block += "arg bytes " + std::to_string(argument.size) + "\n";
        }
        if (!usable) continue;
        block += globals;
        text += block;
        ++emitted;
    }
    if (diagnostics != nullptr) *diagnostics = messages.str();
    return emitted == 0 ? std::string{} : text;
}

}

#pragma once

#include "cumetal/ir/ir.h"

#include <string>
#include <string_view>

namespace cumetal::ptx {

std::string build_ir_abi_sidecar(const ir::Module& module,
                                 const std::string& requested_entry,
                                 std::string_view ptx_source = {},
                                 std::string* diagnostics = nullptr);

}

// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/session/outputs.h"

#include <filesystem>
#include <string>

namespace ernie::bench {

/// Writes a contact sheet of each var of `outputs` to `directory`, as
/// "<stem>_<var>.png": up to 16 views spread through a batch, 4 wide.
void WriteSheets(const std::filesystem::path& directory,
                 const std::string& stem,
                 const RenderProductSetOutputs& outputs);

}  // namespace ernie::bench

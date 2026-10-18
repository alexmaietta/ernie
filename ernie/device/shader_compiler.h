// SPDX-License-Identifier: BSD-3-Clause
/// @file shader_compiler.h
/// @brief Slang modules compiled at run time: to SPIR-V for Vulkan, to WGSL for WebGPU.
///
/// References:
/// - Y. He, K. Fatahalian and T. Foley, "Slang: language mechanisms for extensible real-time
///   shading systems", ACM Transactions on Graphics 37(4), 141, 2018.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// @brief A Slang module: its name, as `import`s name it, and its source.
struct ShaderModule {
    std::string name;
    std::string source;
};

/// @brief Compiles Slang modules, built-in kernels and users' models alike. Each compiler has
/// its own Slang session, started at its first compile; one thread compiles at a time.
class ShaderCompiler {
  public:
    ShaderCompiler();
    ~ShaderCompiler();
    ShaderCompiler(const ShaderCompiler&) = delete;
    ShaderCompiler& operator=(const ShaderCompiler&) = delete;

    /// @brief Compiles `entryPoint` of the last module to SPIR-V 1.6.
    /// @param modules Imported modules before the modules importing them; the last holds the
    /// entry point.
    /// @throws std::invalid_argument with Slang's diagnostics if a module fails or has no such
    /// entry point.
    std::vector<std::uint32_t> Spirv(std::span<const ShaderModule> modules,
                                     std::string_view entryPoint) const;
    /// @brief Compiles `entryPoint` of the last module to WGSL, as `Spirv` does.
    std::string Wgsl(std::span<const ShaderModule> modules, std::string_view entryPoint) const;

  private:
    struct Session;
    std::unique_ptr<Session> session_;
};

}  // namespace ernie

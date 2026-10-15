// SPDX-License-Identifier: BSD-3-Clause
/// @file shader_compiler.h
/// @brief Run-time Slang compilation to SPIR-V and WGSL.
#pragma once

#include "ernie/util/shader_source.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ernie {

/// @brief Orders modules so each loads after the modules it imports.
/// @param modules The program's modules; the entry point's is last.
/// @param library Modules to add when something imports them and
///                `modules` does not give them.
/// @return `modules` plus the library modules they import, each after
///         its imports. A name in `modules` overrides the library's.
/// @note Imports found in neither list are left for Slang to report.
std::vector<ShaderModuleSource>
ResolveImports(std::span<const ShaderModuleSource> modules,
               std::span<const ShaderModuleSource> library);

/// @brief Compiles Slang modules at run time.
///
/// Shaders are composed from built-in and user modules, such as
/// custom reflectance models. A module's `import`s are found among
/// the modules given to a compile, then in the compiler's library.
///
/// References:
/// - Y. He, K. Fatahalian and T. Foley, "Slang: language mechanisms
///   for extensible real-time shading systems", ACM Transactions on
///   Graphics 37(4), 141, 2018.
///   https://doi.org/10.1145/3197517.3201380
class ShaderCompiler {
  public:
    /// @brief Makes a compiler; Slang itself starts at the first
    /// compile the disk cache does not answer.
    /// @param library Modules any compile may import, such as the
    ///                built-in kernels (`shaders::Modules()`).
    explicit ShaderCompiler(
        std::vector<ShaderModuleSource> library = {});
    ~ShaderCompiler();
    ShaderCompiler(const ShaderCompiler&) = delete;
    ShaderCompiler& operator=(const ShaderCompiler&) = delete;

    /// @brief Compiles an entry point to SPIR-V 1.6.
    ///
    /// Results are cached on disk, keyed by the Slang version, the
    /// entry point and every module's name and source: in
    /// `ERNIE_SHADER_CACHE` (empty or "0" turns it off), else
    /// `$XDG_CACHE_HOME/ernie/shaders` or `~/.cache/ernie/shaders`.
    /// @param modules    The program's modules; the last one holds
    ///                   `entryPoint`.
    /// @param entryPoint The entry point's name.
    /// @return The SPIR-V words.
    /// @throws std::invalid_argument With Slang's diagnostics if a
    ///         module fails or the entry point is missing.
    /// @throws std::runtime_error If Slang cannot start.
    std::vector<std::uint32_t>
    CompileSpirv(std::span<const ShaderModuleSource> modules,
                 const std::string& entryPoint) const;

    /// @return How many `CompileSpirv` calls the disk cache answered.
    std::size_t CacheHits() const;

    /// @brief Compiles an entry point to WGSL, for a future WebGPU
    /// backend.
    /// @param modules    As for `CompileSpirv`.
    /// @param entryPoint The entry point's name.
    /// @return The WGSL source.
    /// @throws std::invalid_argument As `CompileSpirv`.
    std::string
    CompileWgsl(std::span<const ShaderModuleSource> modules,
                const std::string& entryPoint) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ernie

// SPDX-License-Identifier: BSD-3-Clause
/// @file shader_compiler.cpp
/// @brief Compiling through Slang's API: a session per compile, modules loaded from source.
#include "ernie/device/shader_compiler.h"

#include "ernie/util/profiler.h"

#include <slang-com-ptr.h>
#include <slang.h>

#include <array>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace ernie {
namespace {

std::string Text(slang::IBlob* blob) {
    return blob != nullptr ? std::string(static_cast<const char*>(blob->getBufferPointer()),
                                         blob->getBufferSize())
                           : std::string();
}

}  // namespace

struct ShaderCompiler::Session {
    Slang::ComPtr<slang::IGlobalSession> global;
    std::mutex mutex;

    /// @return The entry point compiled for the target, as Slang's output blob.
    Slang::ComPtr<slang::IBlob> Compile(std::span<const ShaderModule> modules,
                                        std::string_view entryPoint, SlangCompileTarget format,
                                        const char* profile) {
        if (modules.empty()) {
            throw std::invalid_argument("Compiling no shader modules");
        }
        const std::lock_guard lock(mutex);
        if (!global && SLANG_FAILED(slang::createGlobalSession(global.writeRef()))) {
            throw std::runtime_error("Cannot start the Slang shader compiler");
        }
        const slang::TargetDesc target{.format = format, .profile = global->findProfile(profile)};
        // Matrices column-major, as glm lays them out, so host structs match without transposes.
        std::array<slang::CompilerOptionEntry, 1> options{};
        options[0].name = slang::CompilerOptionName::MatrixLayoutColumn;
        options[0].value.intValue0 = 1;
        const slang::SessionDesc description{.targets = &target,
                                             .targetCount = 1,
                                             .compilerOptionEntries = options.data(),
                                             .compilerOptionEntryCount =
                                                 static_cast<std::uint32_t>(options.size())};
        Slang::ComPtr<slang::ISession> session;
        if (SLANG_FAILED(global->createSession(description, session.writeRef()))) {
            throw std::runtime_error("Cannot create a Slang session");
        }

        slang::IModule* last = nullptr;
        for (const ShaderModule& module : modules) {
            Slang::ComPtr<slang::IBlob> diagnostics;
            const std::string path = module.name + ".slang";
            last = session->loadModuleFromSourceString(
                module.name.c_str(), path.c_str(), module.source.c_str(), diagnostics.writeRef());
            if (last == nullptr) {
                throw std::invalid_argument("Shader module " + module.name +
                                            " failed to compile:\n" + Text(diagnostics));
            }
        }
        const std::string entry(entryPoint);
        Slang::ComPtr<slang::IEntryPoint> found;
        if (SLANG_FAILED(last->findEntryPointByName(entry.c_str(), found.writeRef()))) {
            throw std::invalid_argument("Shader module " + modules.back().name +
                                        " has no entry point " + entry);
        }
        std::array<slang::IComponentType*, 2> parts{last, found};
        Slang::ComPtr<slang::IComponentType> program;
        Slang::ComPtr<slang::IComponentType> linked;
        Slang::ComPtr<slang::IBlob> code;
        Slang::ComPtr<slang::IBlob> diagnostics;
        const bool failed =
            SLANG_FAILED(session->createCompositeComponentType(
                parts.data(), parts.size(), program.writeRef(), diagnostics.writeRef())) ||
            SLANG_FAILED(program->link(linked.writeRef(), diagnostics.writeRef())) ||
            SLANG_FAILED(linked->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef()));
        if (failed) {
            throw std::invalid_argument("Entry point " + entry + " of " + modules.back().name +
                                        " failed to compile:\n" + Text(diagnostics));
        }
        return code;
    }
};

ShaderCompiler::ShaderCompiler() : session_(std::make_unique<Session>()) {}

ShaderCompiler::~ShaderCompiler() = default;

std::vector<std::uint32_t> ShaderCompiler::Spirv(std::span<const ShaderModule> modules,
                                                 std::string_view entryPoint) const {
    ERNIE_ZONE("Compile Slang to SPIR-V");
    const Slang::ComPtr<slang::IBlob> code =
        session_->Compile(modules, entryPoint, SLANG_SPIRV, "spirv_1_6");
    std::vector<std::uint32_t> words(code->getBufferSize() / 4);
    std::memcpy(words.data(), code->getBufferPointer(), words.size() * 4);
    return words;
}

std::string ShaderCompiler::Wgsl(std::span<const ShaderModule> modules,
                                 std::string_view entryPoint) const {
    ERNIE_ZONE("Compile Slang to WGSL");
    return Text(session_->Compile(modules, entryPoint, SLANG_WGSL, "wgsl").get());
}

}  // namespace ernie

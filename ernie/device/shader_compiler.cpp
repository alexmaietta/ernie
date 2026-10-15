// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/shader_compiler.h"

#include "ernie/util/disk_cache.h"
#include "ernie/util/log.h"

#include <slang-com-ptr.h>
#include <slang.h>

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace ernie {

namespace {

namespace fs = std::filesystem;

std::string Text(slang::IBlob* blob) {
    return blob != nullptr
               ? std::string(static_cast<const char*>(
                                 blob->getBufferPointer()),
                             blob->getBufferSize())
               : std::string();
}

/// Bumped when what the cache stores, or how it keys it, changes.
constexpr std::string_view kCacheFormat = "ernie-spirv-1";
constexpr std::uint32_t kSpirvMagic = 0x07230203;

/// The SPIR-V cached at `path`, if it is there and whole.
std::optional<std::vector<std::uint32_t>>
ReadCached(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return std::nullopt;
    }
    const std::streamsize size = file.tellg();
    if (size < 4 || size % 4 != 0) {
        return std::nullopt;
    }
    std::vector<std::uint32_t> words(static_cast<std::size_t>(size) /
                                     4);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(words.data()), size) ||
        words[0] != kSpirvMagic) {
        return std::nullopt;
    }
    return words;
}

/// Writes `words` to `path` through a temporary file renamed into
/// place, so a reader never sees half a file; failures leave no cache
/// entry and are otherwise ignored.
void WriteCached(const fs::path& path,
                 const std::vector<std::uint32_t>& words) {
    static std::atomic<std::uint64_t> serial{0};
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    if (error) {
        return;
    }
#if defined(__unix__) || defined(__APPLE__)
    const long process = static_cast<long>(getpid());
#else
    const long process = 0;
#endif
    fs::path temporary = path;
    temporary += ".tmp" + std::to_string(process) + "_" +
                 std::to_string(serial.fetch_add(1));
    {
        std::ofstream file(temporary, std::ios::binary);
        file.write(reinterpret_cast<const char*>(words.data()),
                   static_cast<std::streamsize>(words.size() * 4));
        if (!file) {
            file.close();
            fs::remove(temporary, error);
            return;
        }
    }
    fs::rename(temporary, path, error);
    if (error) {
        fs::remove(temporary, error);
    }
}

std::string_view TrimLeft(std::string_view text) {
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }
    return text;
}

/// @brief The module names a Slang source imports.
/// @param source Slang source text.
/// @return The names of its `import` and `__exported import` lines.
std::vector<std::string_view> ImportsOf(std::string_view source) {
    constexpr std::string_view kExported = "__exported";
    constexpr std::string_view kImport = "import";
    std::vector<std::string_view> names;
    std::size_t at = 0;
    while (at < source.size()) {
        std::size_t end = source.find('\n', at);
        if (end == std::string_view::npos) {
            end = source.size();
        }
        std::string_view line = TrimLeft(source.substr(at, end - at));
        at = end + 1;
        if (line.starts_with(kExported)) {
            line = TrimLeft(line.substr(kExported.size()));
        }
        if (!line.starts_with(kImport) ||
            line.size() == kImport.size() ||
            !std::isspace(
                static_cast<unsigned char>(line[kImport.size()]))) {
            continue;
        }
        line = TrimLeft(line.substr(kImport.size()));
        std::size_t length = 0;
        while (
            length < line.size() &&
            (std::isalnum(static_cast<unsigned char>(line[length])) ||
             line[length] == '_')) {
            ++length;
        }
        if (length > 0) {
            names.push_back(line.substr(0, length));
        }
    }
    return names;
}

}  // namespace

std::vector<ShaderModuleSource>
ResolveImports(std::span<const ShaderModuleSource> modules,
               std::span<const ShaderModuleSource> library) {
    std::unordered_map<std::string_view, const ShaderModuleSource*>
        available;
    for (const ShaderModuleSource& module : library) {
        available[module.name] = &module;
    }
    for (const ShaderModuleSource& module : modules) {
        available[module.name] = &module;
    }
    std::vector<ShaderModuleSource> ordered;
    std::unordered_set<std::string_view> seen;
    // Depth first: a module goes in after everything it imports.
    const auto visit = [&](const auto& self,
                           const ShaderModuleSource& module) -> void {
        if (!seen.insert(module.name).second) {
            return;
        }
        for (const std::string_view name : ImportsOf(module.source)) {
            if (const auto found = available.find(name);
                found != available.end()) {
                self(self, *found->second);
            }
        }
        ordered.push_back(module);
    };
    for (const ShaderModuleSource& module : modules) {
        visit(visit, *available.at(module.name));
    }
    return ordered;
}

struct ShaderCompiler::Impl {
    /// Created at the first compile the cache does not answer: it
    /// loads Slang's core module, a good part of a second.
    mutable Slang::ComPtr<slang::IGlobalSession> global;
    mutable std::mutex mutex;
    // Compiled SPIR-V kept between runs.
    std::optional<fs::path> cache =
        CacheDirectory("ERNIE_SHADER_CACHE", "shaders");
    mutable std::atomic<std::size_t> cacheHits{0};
    /// Modules any compile may import.
    std::vector<ShaderModuleSource> library;

    /// `global`, created if need be; under `mutex`.
    slang::IGlobalSession& Global() const {
        if (!global && SLANG_FAILED(slang::createGlobalSession(
                           global.writeRef()))) {
            throw std::runtime_error(
                "Cannot initialise the Slang shader compiler");
        }
        return *global;
    }

    /// @brief Compiles `entryPoint` of module `entryModule`.
    /// @param modules     Every module, each after its imports.
    /// @param entryModule The name of the module with the entry
    ///                    point.
    /// @param entryPoint  The entry point's name.
    /// @param format      The target language.
    /// @param profile     The target profile, or null.
    /// @return The generated code.
    Slang::ComPtr<slang::IBlob>
    Compile(std::span<const ShaderModuleSource> modules,
            const std::string& entryModule,
            const std::string& entryPoint, SlangCompileTarget format,
            const char* profile) const {
        std::lock_guard lock(mutex);
        slang::IGlobalSession& slang = Global();
        slang::TargetDesc target{};
        target.format = format;
        if (profile != nullptr) {
            target.profile = slang.findProfile(profile);
        }
        slang::CompilerOptionEntry matrixLayout{};
        matrixLayout.name =
            slang::CompilerOptionName::MatrixLayoutColumn;
        matrixLayout.value.intValue0 = 1;
        slang::SessionDesc sessionDesc{};
        sessionDesc.targets = &target;
        sessionDesc.targetCount = 1;
        sessionDesc.compilerOptionEntries = &matrixLayout;
        sessionDesc.compilerOptionEntryCount = 1;
        Slang::ComPtr<slang::ISession> session;
        if (SLANG_FAILED(slang.createSession(sessionDesc,
                                             session.writeRef()))) {
            throw std::runtime_error("Cannot create a Slang session");
        }

        slang::IModule* entryOwner = nullptr;
        for (const ShaderModuleSource& module : modules) {
            Slang::ComPtr<slang::IBlob> diagnostics;
            const std::string path = module.name + ".slang";
            slang::IModule* loaded =
                session->loadModuleFromSourceString(
                    module.name.c_str(), path.c_str(),
                    module.source.c_str(), diagnostics.writeRef());
            if (loaded == nullptr) {
                throw std::invalid_argument(
                    "Shader module " + module.name +
                    " failed to compile:\n" + Text(diagnostics));
            }
            if (module.name == entryModule) {
                entryOwner = loaded;
            }
        }

        Slang::ComPtr<slang::IEntryPoint> entry;
        if (entryOwner == nullptr ||
            SLANG_FAILED(entryOwner->findEntryPointByName(
                entryPoint.c_str(), entry.writeRef()))) {
            throw std::invalid_argument(
                "Shader module " + entryModule +
                " has no entry point " + entryPoint);
        }
        slang::IComponentType* parts[] = {entryOwner, entry};
        Slang::ComPtr<slang::IComponentType> program;
        Slang::ComPtr<slang::IBlob> diagnostics;
        if (SLANG_FAILED(session->createCompositeComponentType(
                parts, 2, program.writeRef(),
                diagnostics.writeRef()))) {
            throw std::invalid_argument(
                "Cannot compose shader program:\n" +
                Text(diagnostics));
        }
        Slang::ComPtr<slang::IComponentType> linked;
        if (SLANG_FAILED(program->link(linked.writeRef(),
                                       diagnostics.writeRef()))) {
            throw std::invalid_argument(
                "Cannot link shader program:\n" + Text(diagnostics));
        }
        Slang::ComPtr<slang::IBlob> code;
        if (SLANG_FAILED(linked->getEntryPointCode(
                0, 0, code.writeRef(), diagnostics.writeRef()))) {
            throw std::invalid_argument("Cannot generate code for " +
                                        entryPoint + ":\n" +
                                        Text(diagnostics));
        }
        return code;
    }
};

ShaderCompiler::ShaderCompiler(
    std::vector<ShaderModuleSource> library)
    : impl_(std::make_unique<Impl>()) {
    impl_->library = std::move(library);
    // Under the banner (`VulkanContext::Create`), once a process.
    static std::atomic<bool> announced{false};
    if (!announced.exchange(true)) {
        log::Info("   Shader cache: " +
                  (impl_->cache
                       ? impl_->cache->string()
                       : std::string("off (ERNIE_SHADER_CACHE)")));
    }
}

ShaderCompiler::~ShaderCompiler() = default;

std::vector<std::uint32_t> ShaderCompiler::CompileSpirv(
    std::span<const ShaderModuleSource> modules,
    const std::string& entryPoint) const {
    constexpr const char* kProfile = "spirv_1_6";
    if (modules.empty()) {
        throw std::invalid_argument("No shader modules to compile");
    }
    const std::vector<ShaderModuleSource> program =
        ResolveImports(modules, impl_->library);
    const auto start = std::chrono::steady_clock::now();
    // As Warp's kernel cache lines: compiles always, cache hits when
    // verbose.
    const auto report = [&](const char* what, bool detail) {
        char line[256];
        std::snprintf(
            line, sizeof(line), "Shader %s:%s %s in %.2f ms",
            modules.back().name.c_str(), entryPoint.c_str(), what,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                .count());
        detail ? log::Detail(line) : log::Info(line);
    };
    // From the cache when this Slang compiled the same sources for
    // the same entry point and target before.
    std::optional<fs::path> cached;
    if (impl_->cache) {
        CacheKey key;
        key.Add(kCacheFormat);
        key.Add(spGetBuildTagString());
        key.Add(kProfile);
        key.Add(entryPoint);
        key.Add(modules.back().name);
        for (const ShaderModuleSource& module : program) {
            key.Add(module.name);
            key.Add(module.source);
        }
        cached = *impl_->cache / (key.Hex() + ".spv");
        if (std::optional<std::vector<std::uint32_t>> words =
                ReadCached(*cached)) {
            ++impl_->cacheHits;
            report("loaded from cache", true);
            return std::move(*words);
        }
    }
    const Slang::ComPtr<slang::IBlob> code =
        impl_->Compile(program, modules.back().name, entryPoint,
                       SLANG_SPIRV, kProfile);
    std::vector<std::uint32_t> words(code->getBufferSize() /
                                     sizeof(std::uint32_t));
    std::memcpy(words.data(), code->getBufferPointer(),
                words.size() * sizeof(std::uint32_t));
    if (cached) {
        WriteCached(*cached, words);
    }
    report("compiled", false);
    return words;
}

std::size_t ShaderCompiler::CacheHits() const {
    return impl_->cacheHits.load();
}

std::string ShaderCompiler::CompileWgsl(
    std::span<const ShaderModuleSource> modules,
    const std::string& entryPoint) const {
    if (modules.empty()) {
        throw std::invalid_argument("No shader modules to compile");
    }
    return Text(impl_->Compile(
        ResolveImports(modules, impl_->library), modules.back().name,
        entryPoint, SLANG_WGSL, nullptr));
}

}  // namespace ernie

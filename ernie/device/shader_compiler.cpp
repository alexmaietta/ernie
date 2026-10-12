// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/shader_compiler.h"

#include "ernie/util/log.h"

#include <slang-com-ptr.h>
#include <slang.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>

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

/// Where compiled SPIR-V is kept between runs: `ERNIE_SHADER_CACHE`
/// (empty or "0" for none), else the user's cache folder.
std::optional<fs::path> CacheDirectory() {
    if (const char* own = std::getenv("ERNIE_SHADER_CACHE")) {
        const std::string_view value(own);
        if (value.empty() || value == "0") {
            return std::nullopt;
        }
        return fs::path(value);
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME");
        xdg != nullptr && *xdg != '\0') {
        return fs::path(xdg) / "ernie" / "shaders";
    }
#if defined(_WIN32)
    if (const char* local = std::getenv("LOCALAPPDATA");
        local != nullptr && *local != '\0') {
        return fs::path(local) / "ernie" / "shaders";
    }
#else
    if (const char* home = std::getenv("HOME");
        home != nullptr && *home != '\0') {
        return fs::path(home) / ".cache" / "ernie" / "shaders";
    }
#endif
    return std::nullopt;
}

/// Two 64-bit FNV-1a hashes from different starts, a 128-bit key for
/// everything a compile depends on; each piece is length-prefixed.
class CacheKey {
  public:
    void Add(std::string_view text) {
        const std::uint64_t size = text.size();
        Bytes(&size, sizeof(size));
        Bytes(text.data(), text.size());
    }

    std::string Hex() const {
        char text[33];
        std::snprintf(text, sizeof(text), "%016llx%016llx",
                      static_cast<unsigned long long>(hashes_[0]),
                      static_cast<unsigned long long>(hashes_[1]));
        return text;
    }

  private:
    void Bytes(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::uint64_t& hash : hashes_) {
            for (std::size_t i = 0; i < size; ++i) {
                hash = (hash ^ bytes[i]) * 0x100000001b3ull;
            }
        }
    }

    std::array<std::uint64_t, 2> hashes_{0xcbf29ce484222325ull,
                                         0x84222325cbf29ce4ull};
};

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

}  // namespace

struct ShaderCompiler::Impl {
    /// Created at the first compile the cache does not answer: it
    /// loads Slang's core module, a good part of a second.
    mutable Slang::ComPtr<slang::IGlobalSession> global;
    mutable std::mutex mutex;
    std::optional<fs::path> cache = CacheDirectory();
    mutable std::atomic<std::size_t> cacheHits{0};

    /// `global`, created if need be; under `mutex`.
    slang::IGlobalSession& Global() const {
        if (!global && SLANG_FAILED(slang::createGlobalSession(
                           global.writeRef()))) {
            throw std::runtime_error(
                "Cannot initialise the Slang shader compiler");
        }
        return *global;
    }

    Slang::ComPtr<slang::IBlob>
    Compile(std::span<const ShaderModuleSource> modules,
            const std::string& entryPoint, SlangCompileTarget format,
            const char* profile) const {
        if (modules.empty()) {
            throw std::invalid_argument(
                "No shader modules to compile");
        }
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

        slang::IModule* last = nullptr;
        for (const ShaderModuleSource& module : modules) {
            Slang::ComPtr<slang::IBlob> diagnostics;
            const std::string path = module.name + ".slang";
            last = session->loadModuleFromSourceString(
                module.name.c_str(), path.c_str(),
                module.source.c_str(), diagnostics.writeRef());
            if (last == nullptr) {
                throw std::invalid_argument(
                    "Shader module " + module.name +
                    " failed to compile:\n" + Text(diagnostics));
            }
        }

        Slang::ComPtr<slang::IEntryPoint> entry;
        if (SLANG_FAILED(last->findEntryPointByName(
                entryPoint.c_str(), entry.writeRef()))) {
            throw std::invalid_argument(
                "Shader module " + modules.back().name +
                " has no entry point " + entryPoint);
        }
        slang::IComponentType* parts[] = {last, entry};
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

ShaderCompiler::ShaderCompiler() : impl_(std::make_unique<Impl>()) {
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
    const auto start = std::chrono::steady_clock::now();
    // As Warp's kernel cache lines: compiles always, cache hits when
    // verbose.
    const auto report = [&](const char* what, bool detail) {
        char line[256];
        std::snprintf(
            line, sizeof(line), "Shader %s:%s %s in %.2f ms",
            modules.empty() ? "" : modules.back().name.c_str(),
            entryPoint.c_str(), what,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                .count());
        detail ? log::Detail(line) : log::Info(line);
    };
    // From the cache when this Slang compiled the same sources for
    // the same entry point and target before.
    std::optional<fs::path> cached;
    if (impl_->cache && !modules.empty()) {
        CacheKey key;
        key.Add(kCacheFormat);
        key.Add(spGetBuildTagString());
        key.Add(kProfile);
        key.Add(entryPoint);
        for (const ShaderModuleSource& module : modules) {
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
        impl_->Compile(modules, entryPoint, SLANG_SPIRV, kProfile);
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
    return Text(
        impl_->Compile(modules, entryPoint, SLANG_WGSL, nullptr));
}

}  // namespace ernie

// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/slang/compiler.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <format>
#include <iterator>
#include <optional>
#include <regex>
#include <vector>

#include <slang-com-ptr.h>
#include <slang.h>

#include "ernie/device/slang/debug.h"
#include "ernie/util/disk_cache.h"
#include "ernie/util/environment.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/text.h"

namespace ernie::device {

namespace {

using Slang::ComPtr;

std::string Text(slang::IBlob* diagnostics) {
    if (diagnostics == nullptr) {
        return {};
    }
    return {static_cast<const char*>(diagnostics->getBufferPointer()),
            diagnostics->getBufferSize()};
}

[[noreturn]] void Fail(std::string_view what, std::string_view label, slang::IBlob* diagnostics) {
    throw Error("{} of program \"{}\" failed:\n{}", what, label, Text(diagnostics));
}

slang::IGlobalSession& GlobalSession() {
    thread_local ComPtr<slang::IGlobalSession> global = [] {
        ComPtr<slang::IGlobalSession> created;
        if (SLANG_FAILED(slang::createGlobalSession(created.writeRef()))) {
            throw Error("Slang failed to start");
        }
        const std::string prelude = std::format("#include \"{}\"\n", kPreludeHeader);
        created->setLanguagePrelude(SLANG_SOURCE_LANGUAGE_CUDA, prelude.c_str());
        return created;
    }();
    return *global;
}

slang::CompilerOptionEntry Enable(slang::CompilerOptionName name) {
    slang::CompilerOptionEntry entry{};
    entry.name = name;
    entry.value.kind = slang::CompilerOptionValueKind::Int;
    entry.value.intValue0 = 1;
    return entry;
}

constexpr std::string_view kExtension = ".slang";

constexpr int kEntryFormat = 5;

std::string FilePath(std::string_view path) {
    std::string file(path);
    if (file.ends_with(kExtension) && file.find('/') == std::string::npos) {
        std::replace(file.begin(), file.end() - ptrdiff_t(kExtension.size()), '.', '/');
    }
    return file;
}

std::optional<std::string_view> FindEmbedded(const KernelLibrary& library, std::string_view file) {
    for (const EmbeddedKernel& embedded : library.files) {
        if (file == embedded.path) {
            return std::string_view(reinterpret_cast<const char*>(embedded.bytes), embedded.size);
        }
    }
    for (const KernelLibrary* used : library.uses) {
        if (const std::optional<std::string_view> found = FindEmbedded(*used, file)) {
            return found;
        }
    }
    return std::nullopt;
}

std::optional<std::string_view> FindFile(const KernelLibrary* library, const SourceFiles& sources,
                                         std::string_view path) {
    const std::string file = FilePath(path);
    if (const std::optional<std::string_view> embedded =
            FindEmbedded(library != nullptr ? *library : kDeviceKernels, file)) {
        return embedded;
    }
    const auto found = sources.find(file);
    return found != sources.end() ? std::optional<std::string_view>(found->second) : std::nullopt;
}

std::string Normal(std::string_view path) {
    std::vector<std::string_view> segments;
    while (!path.empty()) {
        const size_t slash = path.find('/');
        const std::string_view segment = path.substr(0, slash);
        if (segment == ".." && !segments.empty()) {
            segments.pop_back();
        } else if (segment != "." && !segment.empty()) {
            segments.push_back(segment);
        }
        path = slash == std::string_view::npos ? std::string_view() : path.substr(slash + 1);
    }
    std::string normal;
    for (const std::string_view segment : segments) {
        normal.append(normal.empty() ? "" : "/").append(segment);
    }
    return normal;
}

class ModuleFiles final : public ISlangFileSystem {
  public:
    ModuleFiles(const KernelLibrary* library, const SourceFiles& sources)
        : m_library(library), m_sources(sources) {}

    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(const SlangUUID& uuid,
                                                          void** object) override {
        *object = castAs(uuid);
        return *object != nullptr ? SLANG_OK : SLANG_E_NO_INTERFACE;
    }

    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override {
        return 1;
    }

    SLANG_NO_THROW uint32_t SLANG_MCALL release() override {
        return 1;
    }

    SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& uuid) override {
        const bool known = uuid == ISlangUnknown::getTypeGuid() ||
                           uuid == ISlangCastable::getTypeGuid() ||
                           uuid == ISlangFileSystem::getTypeGuid();
        return known ? static_cast<ISlangFileSystem*>(this) : nullptr;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(const char* path, ISlangBlob** blob) override {
        std::string file = FilePath(Normal(path));
        const std::optional<std::string_view> found = FindFile(m_library, m_sources, file);
        if (!found) {
            return SLANG_E_NOT_FOUND;
        }
        if (std::ranges::find(m_read, file, &std::pair<std::string, std::string>::first) ==
            m_read.end()) {
            m_read.emplace_back(std::move(file), Digest(*found));
        }
        *blob = slang_createBlob(found->data(), found->size());
        return SLANG_OK;
    }

    std::vector<std::pair<std::string, std::string>> Read() const {
        return m_read;
    }

  private:
    const KernelLibrary* m_library;
    const SourceFiles& m_sources;
    std::vector<std::pair<std::string, std::string>> m_read;
};

void Reflect(slang::EntryPointReflection& reflected, slang::ProgramLayout& layout,
             std::string_view entry, std::string_view label, KernelCode& kernel) {
    SlangUInt groupSize[3] = {};
    reflected.getComputeThreadGroupSize(3, groupSize);
    kernel.shape.groupSize = {uint32_t(std::max<SlangUInt>(groupSize[0], 1)),
                              uint32_t(std::max<SlangUInt>(groupSize[1], 1)),
                              uint32_t(std::max<SlangUInt>(groupSize[2], 1))};

    int uniforms = 0;
    for (unsigned i = 0; i < reflected.getParameterCount(); ++i) {
        slang::VariableLayoutReflection* parameter = reflected.getParameterByIndex(i);
        const char* semantic = parameter->getSemanticName();
        if (semantic == nullptr) {
            ++uniforms;
            kernel.shape.paramBytes = uint32_t(parameter->getTypeLayout()->getStride());
        } else if (const std::string name = Lowercase(semantic);
                   name == "sv_dispatchthreadid" || name == "sv_groupid") {
            kernel.shape.foldable = false;
        }
    }
    if (uniforms > 1) {
        throw Error("kernel {} of program \"{}\" takes {} uniform parameters; kernels take one "
                    "struct",
                    entry, label, uniforms);
    }

    for (SlangUInt i = 0; i < layout.getHashedStringCount(); ++i) {
        size_t length = 0;
        const char* text = layout.getHashedString(i, &length);
        kernel.strings.emplace_back(text, length);
    }
}

void AdaptCudaSource(Backend backend, std::string_view entry, std::string_view label,
                     std::string& code) {
    static const std::regex kernels(R"((extern "C" __global__ void \w+\([^)]*\)\s*\{))");
    if (!std::regex_search(code, kernels)) {
        throw Error("Slang no longer declares kernel {} of program \"{}\" as the CUDA and HIP "
                    "backends expect; update AdaptCudaSource for this Slang version",
                    entry, label);
    }
    code = std::regex_replace(code, kernels, "$1\n    if (ernie_outside_dispatch()) { return; }");
    if (backend == Backend::Hip) {
        static const std::regex globals(R"(extern "C" (__constant__ [^;]*;))");
        const bool reached = code.find("SLANG_globalParams") != std::string::npos;
        if (reached && !std::regex_search(code, globals)) {
            throw Error("Slang no longer declares the globals of kernel {} of program \"{}\" as "
                        "the HIP backend expects; update AdaptCudaSource for this Slang version",
                        entry, label);
        }
        code = std::regex_replace(code, globals, R"(extern "C" { $1 })");
    }
}

ComPtr<slang::ISession> CreateSession(slang::IGlobalSession& global, Backend backend,
                                      Traversal traversal, Sampling sampling, bool debugInfo,
                                      ModuleFiles& files, std::string_view label,
                                      bool narrowWaves) {
    slang::TargetDesc target{};
    std::vector<slang::CompilerOptionEntry> options;
    switch (backend) {
    case Backend::Vulkan:
        target.format = SLANG_SPIRV;
        target.profile = global.findProfile("spirv_1_6");
        options.push_back(Enable(slang::CompilerOptionName::GLSLForceScalarLayout));
        options.push_back(Enable(slang::CompilerOptionName::VulkanUseEntryPointName));
        break;
    case Backend::Cuda:
    case Backend::Hip:
        target.format = SLANG_CUDA_SOURCE;
        break;
    case Backend::Metal:
    case Backend::Cpu:
        throw UnsupportedError("{} has no Slang target yet", BackendName(backend));
    }
    if (const std::string level = Environment("ERNIE_EXP_OPT"); !level.empty()) {
        slang::CompilerOptionEntry entry = Enable(slang::CompilerOptionName::Optimization);
        entry.value.intValue0 = std::stoi(level);
        options.push_back(entry);
    }
    if (Environment("ERNIE_EXP_MINOPT") == "1") {
        options.push_back(Enable(slang::CompilerOptionName::MinimumSlangOptimization));
    }
    if (Environment("ERNIE_EXP_SKIPVAL") == "1") {
        options.push_back(Enable(slang::CompilerOptionName::SkipSPIRVValidation));
    }
    if (debugInfo) {
        slang::CompilerOptionEntry lines = Enable(slang::CompilerOptionName::DebugInformation);
        lines.value.intValue0 = SLANG_DEBUG_INFO_LEVEL_STANDARD;
        options.push_back(lines);
    }
    target.compilerOptionEntries = options.data();
    target.compilerOptionEntryCount = uint32_t(options.size());

    const auto flag = [](bool on) { return on ? "1" : "0"; };
    const slang::PreprocessorMacroDesc macros[] = {
        {"ERNIE_HIP", flag(backend == Backend::Hip)},
        {"ERNIE_VULKAN", flag(backend == Backend::Vulkan)},
        {"ERNIE_HARDWARE_TRAVERSAL", flag(traversal == Traversal::Hardware)},
        {"ERNIE_HARDWARE_TEXTURES",
         flag(sampling == Sampling::Hardware && backend == Backend::Vulkan)},
        {"ERNIE_OBJECT_TEXTURES",
         flag(sampling == Sampling::Hardware && backend != Backend::Vulkan)},
        {"ERNIE_NARROW_WAVES", flag(narrowWaves)},
    };

    slang::SessionDesc desc{};
    desc.targets = &target;
    desc.targetCount = 1;
    desc.fileSystem = &files;
    const char* root = ".";
    desc.searchPaths = &root;
    desc.searchPathCount = 1;
    desc.preprocessorMacros = macros;
    desc.preprocessorMacroCount = uint32_t(std::size(macros));
    ComPtr<slang::ISession> session;
    if (SLANG_FAILED(global.createSession(desc, session.writeRef()))) {
        throw Error("Slang failed to start a session for program \"{}\"", label);
    }
    return session;
}

struct KernelParts {
    std::vector<slang::IComponentType*> components;
    ComPtr<slang::IEntryPoint> entryPoint;
};

KernelParts LoadParts(slang::ISession& session, std::span<const std::string> modules,
                      std::string_view entry, std::string_view label,
                      ComPtr<slang::IBlob>& diagnostics) {
    KernelParts loaded;
    for (const std::string& name : modules) {
        slang::IModule* module = session.loadModule(name.c_str(), diagnostics.writeRef());
        if (module == nullptr) {
            Fail(std::format("Compiling module {}", name), label, diagnostics);
        }
        loaded.components.push_back(module);
        if (!loaded.entryPoint) {
            module->findEntryPointByName(std::string(entry).c_str(), loaded.entryPoint.writeRef());
        }
    }
    if (!loaded.entryPoint) {
        throw Error("program \"{}\" has no compute entry point {}", label, entry);
    }
    loaded.components.push_back(loaded.entryPoint);

    const std::string named =
        std::format("export static const uint kKernelHash = {};", SlangStringHash(entry));
    slang::IModule* kernelHash = session.loadModuleFromSourceString(
        "ernie_kernel_hash", "ernie_kernel_hash.slang", named.c_str(), diagnostics.writeRef());
    if (kernelHash == nullptr) {
        Fail("Naming the kernel", label, diagnostics);
    }
    loaded.components.push_back(kernelHash);
    return loaded;
}

struct LinkedKernel {
    ComPtr<slang::IComponentType> composite;
    ComPtr<slang::IComponentType> linked;
    ComPtr<slang::IBlob> code;
};

LinkedKernel Link(slang::ISession& session, std::span<slang::IComponentType* const> components,
                  std::string_view entry, std::string_view label,
                  ComPtr<slang::IBlob>& diagnostics) {
    LinkedKernel kernel;
    const std::string what = std::format("Compiling kernel {}", entry);
    if (SLANG_FAILED(session.createCompositeComponentType(
            components.data(), SlangInt(components.size()), kernel.composite.writeRef(),
            diagnostics.writeRef())) ||
        SLANG_FAILED(kernel.composite->link(kernel.linked.writeRef(), diagnostics.writeRef())) ||
        SLANG_FAILED(kernel.linked->getEntryPointCode(0, 0, kernel.code.writeRef(),
                                                      diagnostics.writeRef()))) {
        Fail(what, label, diagnostics);
    }
    if (const std::string warnings = Text(diagnostics); !warnings.empty()) {
        Log(LogLevel::Debug, "{} of program \"{}\": {}", what, label, warnings);
    }
    return kernel;
}

KernelCode CompileWithSlang(Backend backend, Traversal traversal, Sampling sampling, bool debugInfo,
                            const KernelLibrary* library, const SourceFiles& sources,
                            std::span<const std::string> modules, std::string_view entry,
                            std::string_view label, bool narrowWaves) {
    slang::IGlobalSession& global = GlobalSession();
    ModuleFiles files(library, sources);
    ComPtr<slang::ISession> session =
        CreateSession(global, backend, traversal, sampling, debugInfo, files, label, narrowWaves);

    ComPtr<slang::IBlob> diagnostics;
    const KernelParts parts = LoadParts(*session, modules, entry, label, diagnostics);
    const LinkedKernel linked = Link(*session, parts.components, entry, label, diagnostics);

    slang::EntryPointReflection* reflected =
        linked.linked->getLayout(0, diagnostics.writeRef())->getEntryPointByIndex(0);
    if (reflected->getStage() != SLANG_STAGE_COMPUTE) {
        throw Error("entry point {} of program \"{}\" isn't a compute shader", entry, label);
    }
    KernelCode kernel{.code = std::string(static_cast<const char*>(linked.code->getBufferPointer()),
                                          linked.code->getBufferSize())};
    if (backend == Backend::Cuda || backend == Backend::Hip) {
        AdaptCudaSource(backend, entry, label, kernel.code);
    }
    Reflect(*reflected, *linked.linked->getLayout(0, diagnostics.writeRef()), entry, label, kernel);
    kernel.files = files.Read();
    return kernel;
}

}

namespace {

void AppendField(std::string& entry, std::string_view text) {
    entry += std::format("{} ", text.size());
    entry += text;
    entry += '\n';
}

template <class T> bool ReadNumber(std::string_view entry, size_t& at, char after, T& value) {
    const char* end = entry.data() + entry.size();
    const auto [next, error] = std::from_chars(entry.data() + at, end, value);
    if (error != std::errc() || next == end || *next != after) {
        return false;
    }
    at = size_t(next - entry.data()) + 1;
    return true;
}

std::optional<std::string_view> ReadField(std::string_view entry, size_t& at) {
    size_t size = 0;
    if (!ReadNumber(entry, at, ' ', size) || at + size >= entry.size()) {
        return std::nullopt;
    }
    const std::string_view field = entry.substr(at, size);
    at += size + 1;
    return field;
}

bool Unchanged(const KernelLibrary* library, const SourceFiles& sources,
               std::span<const std::pair<std::string, std::string>> files) {
    return std::ranges::all_of(files, [&](const auto& file) {
        const std::optional<std::string_view> source = FindFile(library, sources, file.first);
        return source && Digest(*source) == file.second;
    });
}

}

std::string KernelCode::Encode() const {
    std::string entry = std::format("{} {} {} {} {} {} {}\n", shape.groupSize.x, shape.groupSize.y,
                                    shape.groupSize.z, shape.paramBytes, shape.foldable ? 1 : 0,
                                    strings.size(), files.size());
    for (const std::string& text : strings) {
        AppendField(entry, text);
    }
    for (const auto& [path, digest] : files) {
        AppendField(entry, path);
        AppendField(entry, digest);
    }
    return entry + code;
}

std::optional<KernelCode> KernelCode::Decode(std::string_view entry) {
    KernelCode kernel;
    size_t strings = 0;
    size_t files = 0;
    int foldable = 0;
    KernelShape& shape = kernel.shape;
    size_t at = 0;
    if (!ReadNumber(entry, at, ' ', shape.groupSize.x) ||
        !ReadNumber(entry, at, ' ', shape.groupSize.y) ||
        !ReadNumber(entry, at, ' ', shape.groupSize.z) ||
        !ReadNumber(entry, at, ' ', shape.paramBytes) || !ReadNumber(entry, at, ' ', foldable) ||
        !ReadNumber(entry, at, ' ', strings) || !ReadNumber(entry, at, '\n', files)) {
        return std::nullopt;
    }
    shape.foldable = foldable != 0;

    for (size_t i = 0; i < strings; ++i) {
        const std::optional<std::string_view> text = ReadField(entry, at);
        if (!text) {
            return std::nullopt;
        }
        kernel.strings.emplace_back(*text);
    }
    for (size_t i = 0; i < files; ++i) {
        const std::optional<std::string_view> path = ReadField(entry, at);
        const std::optional<std::string_view> digest = path ? ReadField(entry, at) : std::nullopt;
        if (!digest) {
            return std::nullopt;
        }
        kernel.files.emplace_back(*path, *digest);
    }

    kernel.code = entry.substr(at);
    return kernel;
}

bool HasModule(const KernelLibrary* library, const SourceFiles& sources, std::string_view name) {
    std::string path(name);
    std::replace(path.begin(), path.end(), '.', '/');
    return FindFile(library, sources, path + std::string(kExtension)).has_value();
}

KernelCode Compile(Backend backend, Traversal traversal, Sampling sampling, bool debugInfo,
                   const KernelLibrary* library, const SourceFiles& sources,
                   std::span<const std::string> modules, std::string_view entry,
                   std::string_view label, const DiskCache& cache, bool narrowWaves) {
    const std::string key = std::format(
        "slang {} entry {} {}{}{}{}{} {} of {}", spGetBuildTagString(), kEntryFormat,
        BackendName(backend), traversal == Traversal::Hardware ? " with hardware traversal" : "",
        sampling == Sampling::Hardware ? " with hardware textures" : "",
        debugInfo ? " with debug info" : "", narrowWaves ? " with narrow waves" : "", entry,
        Join({modules.begin(), modules.end()}));

    std::optional<KernelCode> kernel;
    if (const std::optional<std::string> cached = cache.Load(key)) {
        kernel = KernelCode::Decode(*cached);
        if (kernel && !Unchanged(library, sources, kernel->files)) {
            kernel.reset();
        }
    }
    if (kernel) {
        kernel->cached = true;
    } else {
        kernel = CompileWithSlang(backend, traversal, sampling, debugInfo, library, sources,
                                  modules, entry, label, narrowWaves);
        cache.Store(key, kernel->Encode());
    }

    kernel->debugInfo = debugInfo;
    return *std::move(kernel);
}

}

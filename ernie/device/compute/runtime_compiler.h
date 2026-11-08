// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

#include "ernie/util/disk_cache.h"
#include "ernie/util/library.h"
#include "ernie/util/version.h"

namespace ernie::device::compute {

struct SourceHeader {
    const char* name;
    std::string_view text;
};

struct RuntimeCompiler {
    using Program = struct CompilerProgram*;

    std::string name;
    std::string library;
    Version version;

    int (*getVersion)(int* major, int* minor) = nullptr;
    const char* (*getErrorString)(int result) = nullptr;
    int (*createProgram)(Program* program, const char* source, const char* name, int headers,
                         const char* const* sources, const char* const* names) = nullptr;
    int (*destroyProgram)(Program* program) = nullptr;
    int (*compileProgram)(Program program, int count, const char* const* options) = nullptr;
    int (*getLogSize)(Program program, size_t* bytes) = nullptr;
    int (*getLog)(Program program, char* log) = nullptr;
    int (*getCodeSize)(Program program, size_t* bytes) = nullptr;
    int (*getCode)(Program program, char* code) = nullptr;

    std::string Compile(std::string_view source, std::span<const SourceHeader> headers,
                        std::span<const std::string> options, std::string_view what) const;

    struct Compiled {
        std::string code;
        bool cached = false;
    };

    Compiled CompileCached(std::string_view source, std::span<const SourceHeader> headers,
                           std::span<const std::string> options, std::string_view what,
                           const DiskCache& cache) const;
};

Loaded<RuntimeCompiler> LoadRuntimeCompiler(std::string_view name, std::string_view prefix,
                                            std::string_view code,
                                            std::initializer_list<const char*> names,
                                            std::initializer_list<const char*> paths,
                                            Version minimum,
                                            const std::filesystem::path& beside = {});

class ModuleCache {
  public:
    template <class Load> std::shared_ptr<void> Find(const std::string& binary, Load load) {
        const std::lock_guard lock(m_mutex);
        std::shared_ptr<void>& module = m_modules[binary];
        if (!module) {
            try {
                module = load();
            } catch (...) {
                m_modules.erase(binary);
                throw;
            }
        }
        return module;
    }

    void Clear() {
        const std::lock_guard lock(m_mutex);
        m_modules.clear();
    }

  private:
    std::mutex m_mutex;
    std::unordered_map<std::string, std::shared_ptr<void>> m_modules;
};

}

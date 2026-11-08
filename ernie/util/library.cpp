// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/library.h"

#include <algorithm>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#ifdef __linux__
#include <link.h>
#endif

#include <fmt/format.h>

namespace ernie {
namespace {

void* OpenHandle(const char* name, bool loadedOnly) {
#ifdef _WIN32
    return loadedOnly ? GetModuleHandleA(name) : LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL | (loadedOnly ? RTLD_NOLOAD : 0));
#endif
}

std::vector<std::string> WithUnversioned(std::initializer_list<const char*> names) {
    std::vector<std::string> all(names.begin(), names.end());
    for (const std::string_view name : names) {
        const size_t suffix = name.find(".so.");
        if (suffix != std::string_view::npos) {
            const std::string unversioned(name.substr(0, suffix + 3));
            if (std::ranges::find(all, unversioned) == all.end()) {
                all.push_back(unversioned);
            }
        }
    }
    return all;
}

}

Library Library::First(const std::vector<std::string>& names, bool loadedOnly) {
    Library library;
    for (const std::string& name : names) {
        if (void* handle = OpenHandle(name.c_str(), loadedOnly)) {
            library.m_handle = handle;
            library.m_name = name;
            break;
        }
    }
    return library;
}

Library Library::Find(std::initializer_list<const char*> names) {
    return First(WithUnversioned(names), true);
}

Library Library::Load(std::initializer_list<const char*> names,
                      std::initializer_list<const char*> paths,
                      const std::filesystem::path& beside) {
    const std::vector<std::string> candidates = WithUnversioned(names);
    if (Library loaded = First(candidates, true)) {
        Log(LogLevel::Debug, "using the {} already in the process", loaded.Name());
        return loaded;
    }

    Library library;
    if (!beside.empty()) {
        std::vector<std::string> besidePaths;
        for (const std::string& name : candidates) {
            besidePaths.push_back((beside / name).string());
        }
        library = First(besidePaths, false);
    }
    if (!library) {
        library = First({names.begin(), names.end()}, false);
    }
    if (!library) {
        library = First({paths.begin(), paths.end()}, false);
    }
    if (library) {
        Log(LogLevel::Debug, "loaded {}", library.Name());
    }
    return library;
}

std::filesystem::path Library::Directory() const {
    if (m_handle == nullptr) {
        return {};
    }
#if defined(_WIN32)
    wchar_t path[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameW(static_cast<HMODULE>(m_handle), path, MAX_PATH);
    return length == 0 || length == MAX_PATH ? std::filesystem::path{}
                                             : std::filesystem::path(path).parent_path();
#elif defined(__linux__)
    link_map* map = nullptr;
    if (dlinfo(m_handle, RTLD_DI_LINKMAP, &map) != 0 || map == nullptr || map->l_name == nullptr) {
        return {};
    }
    return std::filesystem::path(map->l_name).parent_path();
#else
    return {};
#endif
}

void* Library::Symbol(const char* symbol) const {
    if (m_handle == nullptr) {
        return nullptr;
    }
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(m_handle), symbol));
#else
    return dlsym(m_handle, symbol);
#endif
}

std::string TooOld(std::string_view what, Version found, Version needed) {
    return fmt::format("{} is version {}; Ernie needs {} or later", what, found.ToString(),
                       needed.ToString());
}

}

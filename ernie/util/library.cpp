// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/library.h"

#include <format>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ernie {
namespace {

void* OpenHandle(const char* name, bool loadedOnly) {
#ifdef _WIN32
    return loadedOnly ? GetModuleHandleA(name) : LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL | (loadedOnly ? RTLD_NOLOAD : 0));
#endif
}

}

Library Library::First(std::initializer_list<const char*> names, bool loadedOnly) {
    Library library;
    for (const char* name : names) {
        if (void* handle = OpenHandle(name, loadedOnly)) {
            library.m_handle = handle;
            library.m_name = name;
            break;
        }
    }
    return library;
}

Library Library::Load(std::initializer_list<const char*> names,
                      std::initializer_list<const char*> paths) {
    if (Library loaded = First(names, true)) {
        Log(LogLevel::Debug, "using the {} already in the process", loaded.Name());
        return loaded;
    }

    Library library = First(names, false);
    if (!library) {
        library = First(paths, false);
    }
    if (library) {
        Log(LogLevel::Debug, "loaded {}", library.Name());
    }
    return library;
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
    return std::format("{} is version {}; Ernie needs {} or later", what, found.ToString(),
                       needed.ToString());
}

}

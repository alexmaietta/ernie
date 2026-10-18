// SPDX-License-Identifier: BSD-3-Clause
/// @file library.cpp
/// @brief Opening shared libraries with dlopen, or LoadLibrary on Windows.
#include "ernie/util/library.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ernie {

bool SharedLibrary::Open(std::initializer_list<const char*> names) {
    for (const char* name : names) {
#if defined(_WIN32)
        handle_ = reinterpret_cast<void*>(LoadLibraryA(name));
#else
        handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
        if (handle_ != nullptr) {
            return true;
        }
    }
    return false;
}

SharedLibrary::~SharedLibrary() {
    if (handle_ != nullptr) {
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(handle_));
#else
        dlclose(handle_);
#endif
    }
}

void* SharedLibrary::Symbol(const char* name) const {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    return dlsym(handle_, name);
#endif
}

}  // namespace ernie

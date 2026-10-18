// SPDX-License-Identifier: BSD-3-Clause
/// @file library.h
/// @brief Shared libraries opened at run time, for optional drivers such as CUDA's that Ernie
/// neither links nor requires.
#pragma once

#include <initializer_list>

namespace ernie {

/// @brief An opened shared library, closed when destroyed.
class SharedLibrary {
  public:
    /// @brief Opens the first of `names` that loads.
    /// @return Whether one did.
    bool Open(std::initializer_list<const char*> names);
    ~SharedLibrary();
    SharedLibrary() = default;
    SharedLibrary(const SharedLibrary&) = delete;
    SharedLibrary& operator=(const SharedLibrary&) = delete;

    explicit operator bool() const {
        return handle_ != nullptr;
    }
    /// @return The address of `name`, or null when the library has none.
    void* Symbol(const char* name) const;
    /// @brief Looks up `name` as a function pointer of type `F`.
    /// @return Whether it was found.
    template <typename F> bool Load(F& function, const char* name) const {
        function = reinterpret_cast<F>(Symbol(name));
        return function != nullptr;
    }

  private:
    void* handle_ = nullptr;
};

}  // namespace ernie

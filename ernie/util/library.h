// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <filesystem>
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/version.h"

namespace ernie {

class Library {
  public:
    Library() = default;

    static Library Load(std::initializer_list<const char*> names,
                        std::initializer_list<const char*> paths = {},
                        const std::filesystem::path& beside = {});

    explicit operator bool() const {
        return m_handle != nullptr;
    }

    const std::string& Name() const {
        return m_name;
    }

    std::filesystem::path Directory() const;

    void* Symbol(const char* symbol) const;

    template <class Function> bool Symbol(const char* symbol, Function& function) const {
        function = reinterpret_cast<Function>(Symbol(symbol));
        return function != nullptr;
    }

  private:
    static Library First(const std::vector<std::string>& names, bool loadedOnly);

    void* m_handle = nullptr;
    std::string m_name;
};

class SymbolResolver {
  public:
    using Lookup = std::function<void*(const char* symbol)>;

    explicit SymbolResolver(Lookup lookup) : m_lookup(std::move(lookup)) {}

    explicit SymbolResolver(Library library)
        : m_lookup([library = std::move(library)](const char* symbol) {
              return library.Symbol(symbol);
          }) {}

    template <class Function> void operator()(const char* symbol, Function& function) {
        function = reinterpret_cast<Function>(m_lookup(symbol));
        if (function == nullptr && m_missing == nullptr) {
            m_missing = symbol;
        }
    }

    const char* Missing() const {
        return m_missing;
    }

  private:
    Lookup m_lookup;
    const char* m_missing = nullptr;
};

template <class Api> using Loaded = std::variant<Api, std::string>;

template <class Api> Loaded<Api> LogUnavailable(std::string_view name, Loaded<Api> loaded) {
    if (const std::string* why = std::get_if<std::string>(&loaded)) {
        Log(LogLevel::Debug, "{} can't be used here: {}", name, *why);
    }
    return loaded;
}

template <class Api> const Api& RequireLoaded(const Loaded<Api>& loaded) {
    if (const Api* api = std::get_if<Api>(&loaded)) {
        return *api;
    }
    throw UnsupportedError(std::get<std::string>(loaded));
}

std::string TooOld(std::string_view what, Version found, Version needed);

}

// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/text.h"

#include <algorithm>
#include <cctype>
#include <cstdint>

#include <fmt/format.h>

namespace ernie {

std::string Lowercase(std::string_view text) {
    std::string lower(text);
    std::ranges::transform(lower, lower.begin(),
                           [](unsigned char c) { return char(std::tolower(c)); });
    return lower;
}

std::string Join(const std::vector<std::string>& parts, std::string_view separator) {
    std::string joined;
    for (const std::string& part : parts) {
        if (!joined.empty()) {
            joined += separator;
        }
        joined += part;
    }
    return joined;
}

std::vector<std::string_view> Split(std::string_view text, char separator) {
    std::vector<std::string_view> parts;
    size_t start = 0;
    for (size_t end = text.find(separator); end != std::string_view::npos;
         end = text.find(separator, start)) {
        parts.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    parts.push_back(text.substr(start));
    return parts;
}

std::pair<std::string_view, std::string_view> SplitLast(std::string_view text, char separator) {
    const size_t at = text.rfind(separator);
    if (at == std::string_view::npos) {
        return {std::string_view(), text};
    }
    return {text.substr(0, at), text.substr(at + 1)};
}

bool IsIdentifier(std::string_view text) {
    const auto letter = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
    };
    const auto letterOrDigit = [&](char c) { return letter(c) || (c >= '0' && c <= '9'); };
    return !text.empty() && letter(text[0]) &&
           std::all_of(text.begin() + 1, text.end(), letterOrDigit);
}

bool Glob(std::string_view pattern, std::string_view text) {
    size_t p = 0;
    size_t t = 0;
    size_t star = std::string_view::npos;
    size_t resume = 0;
    while (t < text.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) {
            ++p;
            ++t;
        } else if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            resume = t;
        } else if (star != std::string_view::npos) {
            p = star + 1;
            t = ++resume;
        } else {
            return false;
        }
    }

    while (p < pattern.size() && pattern[p] == '*') {
        ++p;
    }
    return p == pattern.size();
}

uint64_t Fnv1a(std::string_view bytes, uint64_t hash) {
    for (const char byte : bytes) {
        hash = (hash ^ uint8_t(byte)) * 0x100000001b3ull;
    }
    return hash;
}

std::string Digest(std::string_view bytes) {
    return fmt::format("{:016x}", Fnv1a(bytes));
}

std::string Replaced(std::string text, std::string_view from, std::string_view to) {
    if (from.empty()) {
        return text;
    }
    for (size_t at = text.find(from); at != std::string::npos; at = text.find(from, at)) {
        text.replace(at, from.size(), to);
        at += to.size();
    }
    return text;
}

std::string ModulePath(std::string_view module) {
    std::string path(module);
    std::ranges::replace(path, '.', '/');
    return path + ".slang";
}

std::string MapOrigins(std::string message,
                       std::span<const std::pair<std::string, std::string>> origins) {
    for (const auto& [generated, origin] : origins) {
        message =
            Replaced(Replaced(std::move(message), "./" + generated, origin), generated, origin);
    }
    return message;
}

}

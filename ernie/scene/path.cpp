// SPDX-License-Identifier: BSD-3-Clause
/// @file path.cpp
/// @brief Path parsing, token interning and the path table.
#include "ernie/scene/path.h"

#include "ernie/util/random.h"
#include "ernie/util/text.h"

#include <algorithm>
#include <stdexcept>

namespace ernie {

namespace {

/// Calls `visit(name)` for each name of an absolute path's text; "/a/" ends in an empty name.
template <typename Visit> void ForEachName(std::string_view text, const Visit& visit) {
    if (text.size() <= 1) {
        return;
    }
    for (std::size_t start = 1; start <= text.size();) {
        const std::size_t end = std::min(text.find('/', start), text.size());
        visit(text.substr(start, end - start));
        start = end + 1;
    }
}

}  // namespace

bool Path::IsValidName(std::string_view name) {
    return IsIdentifier(name);
}

Path::Path(std::string_view text) {
    if (text.empty() || text.front() != '/') {
        throw std::invalid_argument("Prim path '" + std::string(text) + "' is not absolute");
    }
    ForEachName(text, [&](std::string_view name) {
        if (!IsValidName(name)) {
            throw std::invalid_argument("Prim path '" + std::string(text) +
                                        "' has an invalid name");
        }
    });
    text_ = text;
}

Path Path::Root() {
    return Path("/");
}

std::string_view Path::Name() const {
    if (text_.size() <= 1) {
        return {};
    }
    return std::string_view(text_).substr(text_.rfind('/') + 1);
}

Path Path::Parent() const {
    if (text_.size() <= 1) {
        return {};
    }
    const std::size_t slash = text_.rfind('/');
    return slash == 0 ? Root() : Path(std::string_view(text_).substr(0, slash));
}

Path Path::Child(std::string_view name) const {
    if (IsEmpty() || !IsValidName(name)) {
        throw std::invalid_argument("Cannot append '" + std::string(name) + "' to '" + text_ + "'");
    }
    Path child;
    child.text_ = IsRoot() ? "/" + std::string(name) : text_ + "/" + std::string(name);
    return child;
}

bool Path::HasPrefix(const Path& ancestor) const {
    if (ancestor.IsEmpty() || IsEmpty()) {
        return false;
    }
    if (ancestor.IsRoot()) {
        return true;
    }
    return text_.starts_with(ancestor.text_) &&
           (text_.size() == ancestor.text_.size() || text_[ancestor.text_.size()] == '/');
}

TokenTable::TokenTable() {
    Intern("");
}

Token TokenTable::Intern(std::string_view text) {
    const auto next = static_cast<Token>(texts_.size());
    const Token token = index_.Insert(std::hash<std::string_view>()(text), next,
                                      [&](Token other) { return texts_[other] == text; });
    if (token == next) {
        texts_.emplace_back(text);
    }
    return token;
}

std::optional<Token> TokenTable::Find(std::string_view text) const {
    return index_.Find(std::hash<std::string_view>()(text),
                       [&](Token other) { return texts_[other] == text; });
}

PathTable::PathTable() : nodes_(1) {}

std::uint64_t PathTable::Hash(PathId parent, Token name) {
    return Mix64(std::uint64_t{parent} << 32 | name);
}

void PathTable::Reserve(std::size_t count) {
    nodes_.reserve(nodes_.size() + count);
    children_.Reserve(children_.Size() + count);
    names_.Reserve(count);
}

PathId PathTable::Child(PathId parent, Token name) {
    const auto next = static_cast<PathId>(nodes_.size());
    const PathId id = children_.Insert(Hash(parent, name), next, [&](PathId other) {
        return nodes_[other].parent == parent && nodes_[other].name == name;
    });
    if (id != next) {
        return id;
    }
    nodes_.push_back({parent, name});
    Node& above = nodes_[parent];
    if (above.firstChild == kNoPath) {
        above.firstChild = id;
    } else {
        nodes_[above.lastChild].nextSibling = id;
    }
    above.lastChild = id;
    return id;
}

PathId PathTable::Intern(const Path& path) {
    if (path.IsEmpty()) {
        throw std::invalid_argument("An empty path names no prim");
    }
    PathId id = kRootPath;
    ForEachName(path.String(), [&](std::string_view name) { id = Child(id, name); });
    return id;
}

std::optional<PathId> PathTable::Find(const Path& path) const {
    if (path.IsEmpty()) {
        return std::nullopt;
    }
    PathId id = kRootPath;
    bool found = true;
    ForEachName(path.String(), [&](std::string_view name) {
        const std::optional<Token> token = found ? names_.Find(name) : std::nullopt;
        const std::optional<PathId> child =
            token ? children_.Find(Hash(id, *token),
                                   [&](PathId other) {
                                       return nodes_[other].parent == id &&
                                              nodes_[other].name == *token;
                                   })
                  : std::nullopt;
        found = child.has_value();
        if (found) {
            id = *child;
        }
    });
    return found ? std::optional(id) : std::nullopt;
}

Path PathTable::ToPath(PathId id) const {
    if (id == kRootPath) {
        return Path::Root();
    }
    std::vector<std::string_view> names;
    for (; id != kRootPath; id = nodes_[id].parent) {
        names.push_back(names_.Text(nodes_[id].name));
    }
    std::string text;
    for (auto name = names.rbegin(); name != names.rend(); ++name) {
        text += '/';
        text += *name;
    }
    return Path(text);
}

bool PathTable::HasPrefix(PathId id, PathId ancestor) const {
    for (; id != kNoPath; id = nodes_[id].parent) {
        if (id == ancestor) {
            return true;
        }
    }
    return false;
}

}  // namespace ernie

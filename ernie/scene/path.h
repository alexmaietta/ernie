// SPDX-License-Identifier: BSD-3-Clause
/// @file path.h
/// @brief Prim paths, interned names, and the scene's table of paths as (parent, name) nodes.
#pragma once

#include "ernie/export.h"
#include "ernie/util/hash_index.h"

#include <compare>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// @brief An absolute prim path such as `/World/Rover/Wheel_FL`, as text.
class ERNIE_API Path {
  public:
    Path() = default;
    /// @param text An absolute path of valid names.
    /// @throws std::invalid_argument If it is not.
    Path(std::string_view text);
    Path(const char* text) : Path(std::string_view(text)) {}
    Path(const std::string& text) : Path(std::string_view(text)) {}

    static Path Root();
    /// @return Whether `name` is a valid prim name: a letter or underscore, then letters, digits
    ///         and underscores.
    static bool IsValidName(std::string_view name);

    const std::string& String() const {
        return text_;
    }
    bool IsEmpty() const {
        return text_.empty();
    }
    bool IsRoot() const {
        return text_ == "/";
    }
    /// @return The last name; empty for the root and the empty path.
    std::string_view Name() const;
    /// @return The parent; empty for the root and the empty path.
    Path Parent() const;
    /// @throws std::invalid_argument If this path is empty or `name` is invalid.
    Path Child(std::string_view name) const;
    /// @return Whether this path is `ancestor` or lies below it.
    bool HasPrefix(const Path& ancestor) const;

    auto operator<=>(const Path&) const = default;

  private:
    std::string text_;
};

/// An interned name; 0 is no name.
using Token = std::uint32_t;

/// @brief Interned strings: each distinct name is stored once and compared as a number.
class TokenTable {
  public:
    TokenTable();
    /// @return The name's token, adding it if new.
    Token Intern(std::string_view text);
    /// @return The name's token, if interned.
    std::optional<Token> Find(std::string_view text) const;
    std::string_view Text(Token token) const {
        return texts_[token];
    }
    void Reserve(std::size_t count) {
        index_.Reserve(count);
    }

  private:
    std::deque<std::string> texts_;
    HashIndex index_;
};

/// An interned path, a node of a `PathTable`; 0 is the root.
using PathId = std::uint32_t;
inline constexpr PathId kRootPath = 0;
inline constexpr PathId kNoPath = std::numeric_limits<PathId>::max();

/// @brief Every path the scene has named, as (parent, name) nodes with their children in order.
/// Paths are never removed, so ids stay valid; prims come and go on them.
class PathTable {
  public:
    PathTable();

    /// @return The path's id, adding it and its ancestors if new.
    PathId Intern(const Path& path);
    /// @return The child's id, adding it if new.
    PathId Child(PathId parent, Token name);
    PathId Child(PathId parent, std::string_view name) {
        return Child(parent, names_.Intern(name));
    }
    /// @return The path's id, if interned.
    std::optional<PathId> Find(const Path& path) const;

    Path ToPath(PathId id) const;
    PathId Parent(PathId id) const {
        return nodes_[id].parent;
    }
    Token Name(PathId id) const {
        return nodes_[id].name;
    }
    PathId FirstChild(PathId id) const {
        return nodes_[id].firstChild;
    }
    PathId NextSibling(PathId id) const {
        return nodes_[id].nextSibling;
    }
    /// Calls `visit(child)` for each child of `id`, in the order they were added.
    template <typename Visit> void ForEachChild(PathId id, const Visit& visit) const {
        for (PathId child = nodes_[id].firstChild; child != kNoPath;
             child = nodes_[child].nextSibling) {
            visit(child);
        }
    }
    /// @return Whether `id` is `ancestor` or lies below it.
    bool HasPrefix(PathId id, PathId ancestor) const;

    std::size_t Size() const {
        return nodes_.size();
    }
    /// Makes room for `count` more paths and as many new names.
    void Reserve(std::size_t count);
    TokenTable& Names() {
        return names_;
    }
    const TokenTable& Names() const {
        return names_;
    }

  private:
    struct Node {
        PathId parent = kNoPath;
        Token name = 0;
        PathId firstChild = kNoPath;
        PathId lastChild = kNoPath;
        PathId nextSibling = kNoPath;
    };
    static std::uint64_t Hash(PathId parent, Token name);

    TokenTable names_;
    std::vector<Node> nodes_;
    HashIndex children_;
};

}  // namespace ernie

template <> struct std::hash<ernie::Path> {
    std::size_t operator()(const ernie::Path& path) const noexcept {
        return std::hash<std::string>()(path.String());
    }
};

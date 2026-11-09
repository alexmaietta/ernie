// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ernie/stage/structure/tokens.h"

namespace ernie {

struct PathShape {
    std::vector<uint32_t> parent;
    std::vector<Token> name;
    std::vector<uint32_t> depth;
    std::vector<std::vector<uint32_t>> children;
    std::unordered_map<uint64_t, uint32_t> lookup;
};

class Paths {
  public:
    static constexpr uint32_t kNone = 0xFFFFFFFF;
    static constexpr uint32_t kRoot = kNone;

    explicit Paths(Tokens& tokens) : m_tokens(&tokens) {}

    Paths(const Paths&) = delete;
    Paths& operator=(const Paths&) = delete;
    Paths(Paths&&) = default;
    Paths& operator=(Paths&&) = default;

    void Insert(std::string_view path, uint32_t prim);
    PathShape Shape(uint32_t source);
    void InsertClones(PathShape shape, std::span<const std::string> targets, uint32_t firstPrim);
    void Erase(uint32_t prim);
    void Move(uint32_t prim, std::string_view path);

    uint32_t Find(std::string_view path) const;
    uint32_t FindParent(std::string_view path) const;
    bool Contains(uint32_t prim) const;

    std::string Path(uint32_t prim) const;
    std::string_view Name(uint32_t prim) const;
    uint32_t Parent(uint32_t prim) const;
    uint32_t Depth(uint32_t prim) const;
    std::vector<uint32_t> Children(uint32_t prim) const;
    std::vector<uint32_t> Subtree(uint32_t prim) const;

    uint32_t Size() const {
        return m_size + m_lazy;
    }

    uint32_t Explicit() const {
        return m_size;
    }

  private:
    struct Node {
        uint32_t parent = kNone;
        Token name;
        uint32_t depth = 0;
        bool used = false;
        std::vector<uint32_t> children;
    };

    struct Block {
        PathShape shape;
        uint32_t firstPrim = 0;
        uint32_t copies = 0;
        std::vector<bool> explicitCopy;
    };

    struct Lazy {
        const Block* block;
        uint32_t copy;
        uint32_t index;
    };

    static uint64_t Key(uint32_t parent, Token name);
    const Node& Get(uint32_t prim) const;
    uint32_t Child(uint32_t parent, std::string_view name) const;
    std::optional<Lazy> Locate(uint32_t prim) const;
    void Materialize(uint32_t prim);
    void Link(uint32_t prim, uint32_t parent, Token name);

    Tokens* m_tokens;
    std::vector<Node> m_nodes;
    std::vector<uint32_t> m_roots;
    std::unordered_map<uint64_t, uint32_t> m_children;
    std::vector<Block> m_blocks;
    std::map<uint32_t, uint32_t> m_blockStarts;
    uint32_t m_size = 0;
    uint32_t m_lazy = 0;
};

}

// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/catalog/paths.h"

#include <algorithm>

#include "ernie/util/error.h"
#include "ernie/util/text.h"

namespace ernie {

namespace {

void CheckPath(std::string_view path) {
    if (path.empty() || path.front() != '/') {
        throw Error("Path \"{}\" isn't absolute", path);
    }
    if (path == "/") {
        throw Error("The root, \"/\", isn't a prim");
    }
    if (path.back() == '/') {
        throw Error("Path \"{}\" ends in /", path);
    }
}

}

uint64_t Paths::Key(uint32_t parent, Token name) {
    return (uint64_t(parent) << 32) | name.id;
}

const Paths::Node& Paths::At(uint32_t prim) const {
    if (prim >= m_nodes.size() || !m_nodes[prim].used) {
        throw Error("No prim {} in the hierarchy", prim);
    }
    return m_nodes[prim];
}

std::optional<Paths::Lazy> Paths::Locate(uint32_t prim) const {
    auto found = m_blockStarts.upper_bound(prim);
    if (prim == kNone || found == m_blockStarts.begin()) {
        return std::nullopt;
    }
    const Block& block = m_blocks[std::prev(found)->second];
    const uint32_t count = uint32_t(block.shape.parent.size());
    const uint32_t offset = prim - block.firstPrim;
    if (offset >= count * block.copies || block.explicitCopy[offset / count]) {
        return std::nullopt;
    }
    return Lazy{&block, offset / count, offset % count};
}

bool Paths::Contains(uint32_t prim) const {
    if (prim < m_nodes.size() && m_nodes[prim].used) {
        return true;
    }
    const std::optional<Lazy> lazy = Locate(prim);
    return lazy && lazy->index > 0;
}

uint32_t Paths::Child(uint32_t parent, std::string_view name) const {
    const Token token = m_tokens->Find(name);
    if (token.id == 0) {
        return kNone;
    }
    if (const std::optional<Lazy> lazy = Locate(parent)) {
        const auto found = lazy->block->shape.lookup.find(Key(lazy->index, token));
        if (found == lazy->block->shape.lookup.end()) {
            return kNone;
        }
        return lazy->block->firstPrim + lazy->copy * uint32_t(lazy->block->shape.parent.size()) +
               found->second;
    }
    const auto found = m_children.find(Key(parent, token));
    return found == m_children.end() ? kNone : found->second;
}

void Paths::Link(uint32_t prim, uint32_t parent, Token name) {
    if (!m_children.emplace(Key(parent, name), prim).second) {
        throw Error("A prim named {} is already under prim {}", m_tokens->Text(name), parent);
    }
    if (prim >= m_nodes.size()) {
        m_nodes.resize(size_t(prim) + 1);
    }
    Node& node = m_nodes[prim];
    node.parent = parent;
    node.name = name;
    node.depth = parent == kRoot ? 0 : m_nodes[parent].depth + 1;
    node.used = true;
    node.children.clear();
    (parent == kRoot ? m_roots : m_nodes[parent].children).push_back(prim);
    ++m_size;
}

void Paths::Materialize(uint32_t prim) {
    const std::optional<Lazy> lazy = Locate(prim);
    if (!lazy) {
        return;
    }
    Block& block = m_blocks[m_blockStarts.at(lazy->block->firstPrim)];
    const uint32_t count = uint32_t(block.shape.parent.size());
    const uint32_t first = block.firstPrim + lazy->copy * count;
    block.explicitCopy[lazy->copy] = true;
    for (uint32_t i = 1; i < count; ++i) {
        Link(first + i, first + block.shape.parent[i], block.shape.name[i]);
    }
    m_lazy -= count - 1;
}

PathShape Paths::Shape(uint32_t source) {
    const std::vector<uint32_t> subtree = Subtree(source);
    PathShape shape;
    std::unordered_map<uint32_t, uint32_t> index;
    for (uint32_t i = 0; i < subtree.size(); ++i) {
        index[subtree[i]] = i;
        const uint32_t parent = i == 0 ? kNone : index.at(Parent(subtree[i]));
        shape.parent.push_back(parent);
        shape.name.push_back(m_tokens->Intern(Name(subtree[i])));
        shape.depth.push_back(i == 0 ? 0 : shape.depth[parent] + 1);
        shape.children.emplace_back();
        if (i > 0) {
            shape.children[parent].push_back(i);
            shape.lookup.emplace(Key(parent, shape.name[i]), i);
        }
    }
    return shape;
}

void Paths::InsertClones(PathShape shape, std::span<const std::string> targets,
                         uint32_t firstPrim) {
    const uint32_t count = uint32_t(shape.parent.size());
    Block block;
    block.shape = std::move(shape);
    block.firstPrim = firstPrim;
    block.copies = uint32_t(targets.size());
    block.explicitCopy.assign(targets.size(), false);
    for (uint32_t copy = 0; copy < targets.size(); ++copy) {
        Insert(targets[copy], firstPrim + copy * count);
    }
    m_blockStarts[firstPrim] = uint32_t(m_blocks.size());
    m_blocks.push_back(std::move(block));
    m_lazy += (count - 1) * uint32_t(targets.size());
}

uint32_t Paths::Find(std::string_view path) const {
    if (path.empty() || path.front() != '/' || path == "/" || path.back() == '/') {
        return kNone;
    }
    uint32_t prim = kRoot;
    for (const std::string_view name : Split(path.substr(1), '/')) {
        prim = Child(prim, name);
        if (prim == kNone) {
            return kNone;
        }
    }
    return prim;
}

uint32_t Paths::FindParent(std::string_view path) const {
    CheckPath(path);
    const auto [parent, name] = SplitLast(path, '/');
    return parent.empty() ? kRoot : Find(parent);
}

void Paths::Insert(std::string_view path, uint32_t prim) {
    CheckPath(path);
    if (prim == kNone) {
        throw Error("Prim index {} is reserved", prim);
    }
    if (Contains(prim)) {
        throw Error("Prim {} is already at \"{}\"", prim, Path(prim));
    }
    const auto [parentPath, text] = SplitLast(path, '/');
    if (!IsIdentifier(text)) {
        throw Error("\"{}\" in path \"{}\" isn't a name: names are letters, digits and _, and "
                    "don't start with a digit",
                    text, path);
    }
    const uint32_t parent = parentPath.empty() ? kRoot : Find(parentPath);
    if (parent == kNone && !parentPath.empty()) {
        throw Error("Path \"{}\" has no parent \"{}\"", path, parentPath);
    }
    if (Find(path) != kNone) {
        throw Error("Path \"{}\" already has a prim", path);
    }
    Materialize(parent);
    Link(prim, parent, m_tokens->Intern(text));
}

void Paths::Erase(uint32_t prim) {
    if (!Contains(prim)) {
        throw Error("No prim {} in the hierarchy", prim);
    }
    Materialize(prim);
    Node& node = m_nodes[prim];
    if (!node.children.empty()) {
        throw Error("Prim \"{}\" has {} children; erase them first", Path(prim),
                    node.children.size());
    }
    std::vector<uint32_t>& siblings =
        node.parent == kRoot ? m_roots : m_nodes[node.parent].children;
    siblings.erase(std::ranges::find(siblings, prim));
    m_children.erase(Key(node.parent, node.name));
    m_nodes[prim] = {};
    --m_size;
}

void Paths::Move(uint32_t prim, std::string_view path) {
    CheckPath(path);
    if (!Contains(prim)) {
        throw Error("No prim {} in the hierarchy", prim);
    }
    const auto [parentPath, text] = SplitLast(path, '/');
    if (!IsIdentifier(text)) {
        throw Error("\"{}\" in path \"{}\" isn't a name: names are letters, digits and _, and "
                    "don't start with a digit",
                    text, path);
    }
    const uint32_t parent = parentPath.empty() ? kRoot : Find(parentPath);
    if (parent == kNone && !parentPath.empty()) {
        throw Error("Path \"{}\" has no parent \"{}\"", path, parentPath);
    }
    for (uint32_t at = parent; at != kRoot; at = Parent(at)) {
        if (at == prim) {
            throw Error("\"{}\" can't move into its own subtree at \"{}\"", Path(prim), path);
        }
    }
    if (Find(path) != kNone) {
        throw Error("Path \"{}\" already has a prim", path);
    }
    Materialize(prim);
    Materialize(parent);
    const Token name = m_tokens->Intern(text);
    Node& node = m_nodes[prim];
    std::vector<uint32_t>& siblings =
        node.parent == kRoot ? m_roots : m_nodes[node.parent].children;
    siblings.erase(std::ranges::find(siblings, prim));
    m_children.erase(Key(node.parent, node.name));
    m_children.emplace(Key(parent, name), prim);
    node.parent = parent;
    node.name = name;
    (parent == kRoot ? m_roots : m_nodes[parent].children).push_back(prim);
    std::vector<uint32_t> stack = {prim};
    while (!stack.empty()) {
        Node& at = m_nodes[stack.back()];
        stack.pop_back();
        at.depth = at.parent == kRoot ? 0 : m_nodes[at.parent].depth + 1;
        stack.insert(stack.end(), at.children.begin(), at.children.end());
    }
}

std::string Paths::Path(uint32_t prim) const {
    std::vector<std::string_view> names;
    for (uint32_t at = prim; at != kRoot; at = Parent(at)) {
        names.push_back(Name(at));
    }
    std::string path;
    for (auto name = names.rbegin(); name != names.rend(); ++name) {
        path += '/';
        path += *name;
    }
    return path;
}

std::string_view Paths::Name(uint32_t prim) const {
    if (const std::optional<Lazy> lazy = Locate(prim); lazy && lazy->index > 0) {
        return m_tokens->Text(lazy->block->shape.name[lazy->index]);
    }
    return m_tokens->Text(At(prim).name);
}

uint32_t Paths::Parent(uint32_t prim) const {
    if (const std::optional<Lazy> lazy = Locate(prim); lazy && lazy->index > 0) {
        const uint32_t count = uint32_t(lazy->block->shape.parent.size());
        return lazy->block->firstPrim + lazy->copy * count + lazy->block->shape.parent[lazy->index];
    }
    return At(prim).parent;
}

uint32_t Paths::Depth(uint32_t prim) const {
    if (const std::optional<Lazy> lazy = Locate(prim); lazy && lazy->index > 0) {
        const uint32_t count = uint32_t(lazy->block->shape.parent.size());
        return At(lazy->block->firstPrim + lazy->copy * count).depth +
               lazy->block->shape.depth[lazy->index];
    }
    return At(prim).depth;
}

std::vector<uint32_t> Paths::Children(uint32_t prim) const {
    if (prim == kRoot) {
        return m_roots;
    }
    if (const std::optional<Lazy> lazy = Locate(prim)) {
        const uint32_t first =
            lazy->block->firstPrim + lazy->copy * uint32_t(lazy->block->shape.parent.size());
        std::vector<uint32_t> children;
        for (const uint32_t child : lazy->block->shape.children[lazy->index]) {
            children.push_back(first + child);
        }
        return children;
    }
    return At(prim).children;
}

std::vector<uint32_t> Paths::Subtree(uint32_t prim) const {
    std::vector<uint32_t> order;
    std::vector<uint32_t> stack = {prim};
    if (prim != kRoot && !Contains(prim)) {
        throw Error("No prim {} in the hierarchy", prim);
    }
    while (!stack.empty()) {
        const uint32_t at = stack.back();
        stack.pop_back();
        if (at != kRoot) {
            order.push_back(at);
        }
        const std::vector<uint32_t> children = Children(at);
        stack.insert(stack.end(), children.rbegin(), children.rend());
    }
    return order;
}

}

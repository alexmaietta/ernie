// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/catalog/catalog.h"

#include <algorithm>
#include <cstring>
#include <ranges>
#include <set>
#include <unordered_map>
#include <utility>

#include "ernie/scene/catalog/packing.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/util/error.h"
#include "ernie/util/span.h"
#include "ernie/util/text.h"

namespace ernie {

std::vector<uint32_t> RowWriter::Prims() const {
    if (!m_prims.empty()) {
        return {m_prims.begin(), m_prims.end()};
    }
    std::vector<uint32_t> prims;
    prims.reserve(m_block.PrimCount());
    for (uint32_t copy = 0; copy < m_block.copies; ++copy) {
        for (uint32_t index = 0; index < m_block.count; ++index) {
            prims.push_back(m_block.Prim(copy, index));
        }
    }
    return prims;
}

const BlockTable& RowWriter::Rows(SchemaId table) const {
    const auto found = std::ranges::find(m_block.tables, table, &BlockTable::table);
    if (found == m_block.tables.end()) {
        throw Error("These prims have no {} data", Info(table).name);
    }
    return *found;
}

uint32_t RowWriter::RowCount(const BlockTable& rows) const {
    return rows.count != kNoRow ? rows.count : rows.perCopy * m_block.copies;
}

void RowWriter::Fill(ColumnRef column, std::span<const std::byte> cell) {
    const BlockTable& rows = Rows(column.schema);
    if (RowCount(rows) > 0) {
        m_catalog.m_store.Fill(column, rows.first, RowCount(rows), cell);
    }
}

void RowWriter::SetToken(ColumnRef column, std::string_view text) {
    const uint32_t id = m_catalog.m_tokens.Intern(text).id;
    Fill(column, std::as_bytes(std::span(&id, 1)));
}

void RowWriter::SetArray(ColumnRef column, std::span<const std::byte> bytes, uint32_t count) {
    if (count == 0) {
        return;
    }
    Store& store = m_catalog.m_store;
    ArrayHeap& arrays = store.Arrays(column.column->type);
    if (bytes.size() != size_t(count) * arrays.ItemBytes()) {
        throw Error("{} bytes for {} items of {}'s {}, whose items are {} bytes", bytes.size(),
                    count, Info(column.schema).name, column.column->field, arrays.ItemBytes());
    }
    const ArrayRef array = arrays.Allocate(count);
    arrays.Write(array, bytes);
    const BlockTable& rows = Rows(column.schema);
    if (RowCount(rows) == 0) {
        return;
    }
    const uint32_t last = rows.first + RowCount(rows);
    for (uint32_t row = rows.first; row < last; ++row) {
        store.SetArray(column, row, row == rows.first ? array : arrays.Share(array));
    }
}

void RowWriter::SetTokens(ColumnRef column, std::span<const std::string> texts) {
    const std::vector<uint32_t> ids = InternTokens(m_catalog.m_tokens, texts);
    SetArray(column, std::as_bytes(std::span(ids)), uint32_t(ids.size()));
}

void RowWriter::SetTargets(ColumnRef column, std::span<const std::string> paths) {
    const std::vector<uint32_t> targets = m_catalog.ResolveTargets(paths);
    m_catalog.MarkTargets(targets);
    SetArray(column, std::as_bytes(std::span(targets)), uint32_t(targets.size()));
    if (std::ranges::find(targets, kNoPrim) == targets.end()) {
        return;
    }
    const BlockTable& rows = Rows(column.schema);
    for (uint32_t copy = 0; copy < m_block.copies; ++copy) {
        for (uint32_t index = 0; index < m_block.count; ++index) {
            const uint32_t prim = m_prims.empty() ? m_block.Prim(copy, index) : m_prims[copy];
            const uint32_t row = m_catalog.m_prims.FindRow(prim, column.schema);
            m_catalog.m_pending.push_back({prim, m_catalog.m_prims.Generation(prim), column,
                                           IsMultipleApi(column.schema) ? rows.first + copy : row,
                                           std::vector<std::string>(paths.begin(), paths.end())});
        }
    }
}

void RowWriter::SetPrimvars(ColumnRef column, std::span<const Primvar> primvars) {
    const std::vector<int32_t> table = PackPrimvars(m_catalog.m_tokens, primvars);
    SetArray(column, std::as_bytes(std::span(table)), uint32_t(table.size()));
}

Catalog::Catalog(Store& store) : m_store(store), m_paths(m_tokens), m_prims(store) {}

uint32_t Catalog::Find(std::string_view path) const {
    return m_paths.Find(path);
}

std::vector<uint32_t> Catalog::Select(std::string_view pattern, SchemaId schema) const {
    if (pattern.empty() || pattern.front() != '/' || pattern == "/") {
        throw Error("Pattern \"{}\" isn't an absolute path", pattern);
    }
    std::vector<uint32_t> level = {Paths::kRoot};
    for (const std::string_view part : Split(pattern.substr(1), '/')) {
        std::vector<uint32_t> next;
        for (const uint32_t parent : level) {
            for (const uint32_t child : m_paths.Children(parent)) {
                if (Glob(part, m_paths.Name(child))) {
                    next.push_back(child);
                }
            }
        }
        level = std::move(next);
    }
    if (schema != SchemaId::None) {
        std::erase_if(level, [&](uint32_t prim) {
            return !IsA(m_prims.At(prim).schema, schema) && !m_prims.HasApi(prim, schema);
        });
    }
    return level;
}

uint32_t Catalog::NewParent(std::string_view path) const {
    if (const uint32_t existing = m_paths.Find(path); existing != Paths::kNone) {
        throw Error("\"{}\" is already a {}", path, Info(m_prims.At(existing).schema).name);
    }
    m_paths.FindParent(path);
    const auto [parentPath, name] = SplitLast(path, '/');
    if (!IsIdentifier(name)) {
        throw Error("\"{}\" in path \"{}\" isn't a name: names are letters, digits and _, and "
                    "don't start with a digit",
                    name, path);
    }
    if (parentPath.empty()) {
        return kNoPrim;
    }
    const uint32_t parent = m_paths.Find(parentPath);
    if (parent == Paths::kNone) {
        throw Error("\"{}\" has no parent \"{}\"", path, parentPath);
    }
    return parent;
}

PrimBlock Catalog::Allocate(SchemaId schema, std::span<const std::string> paths, uint64_t apis,
                            bool attachable) {
    std::set<std::string_view> seen;
    std::vector<uint32_t> parents;
    for (const std::string& path : paths) {
        if (!seen.insert(path).second) {
            throw Error("\"{}\" is defined twice", path);
        }
        parents.push_back(NewParent(path));
    }
    const PrimBlock block =
        m_prims.Add({{schema}, {kNoPrim}, {apis}}, parents, kGlobalWorld, attachable);
    for (uint32_t copy = 0; copy < block.copies; ++copy) {
        const uint32_t prim = block.Prim(copy, 0);
        try {
            m_paths.Insert(paths[copy], prim);
        } catch (...) {
            for (uint32_t undo = 0; undo < copy; ++undo) {
                m_paths.Erase(block.Prim(undo, 0));
            }
            for (uint32_t undo = 0; undo < block.copies; ++undo) {
                m_prims.Remove(block.Prim(undo, 0));
            }
            throw;
        }
        Logged(prim, true);
    }
    ++m_epoch;
    return block;
}

PrimBlock Catalog::Define(SchemaId schema, std::span<const std::string> paths) {
    return Define(schema, paths, [&](RowWriter& writer) { WriteFallbacks(schema, writer); });
}

void Catalog::RemoveBlock(const PrimBlock& block) {
    for (uint32_t prim = block.firstPrim + block.PrimCount(); prim-- > block.firstPrim;) {
        if (m_prims.Alive(prim)) {
            RemovePrim(prim);
        }
    }
}

void Catalog::Logged(uint32_t prim, bool added, SchemaId api) {
    m_log.push_back(
        {m_store.Stamp(), prim, m_prims.Generation(prim), m_prims.At(prim).world, added, api});
}

std::vector<uint32_t> Catalog::Applicable(std::span<const uint32_t> prims, SchemaId api,
                                          bool multiple) const {
    if (!IsApi(api)) {
        throw Error("{} isn't an API schema", Info(api).name);
    }
    if (IsMultipleApi(api) && !multiple) {
        throw Error("{} is applied many times, so it needs an instance name", Info(api).name);
    }
    if (!IsMultipleApi(api) && multiple) {
        throw Error("{} is applied once, without an instance name", Info(api).name);
    }
    std::vector<uint32_t> targets;
    std::set<uint32_t> seen;
    for (const uint32_t prim : prims) {
        if (!m_prims.Alive(prim)) {
            throw Error("Prim {} was removed", prim);
        }
        const SchemaId schema = m_prims.At(prim).schema;
        if (!AppliesTo(api, schema) && !Includes(schema, api)) {
            throw Error("{} doesn't apply to {}, a {}", Info(api).name, m_paths.Path(prim),
                        Info(schema).name);
        }
        if ((multiple || !m_prims.HasApi(prim, api)) && seen.insert(prim).second) {
            targets.push_back(prim);
        }
    }
    return targets;
}

void Catalog::Apply(std::span<const uint32_t> prims, SchemaId api) {
    Apply(prims, api, [&](RowWriter& writer) { WriteFallbacks(api, writer); });
}

void Catalog::Unapply(std::span<const uint32_t> prims, SchemaId api,
                      std::optional<std::string_view> instance) {
    if (!IsApi(api)) {
        throw Error("{} isn't an API schema", Info(api).name);
    }
    if (instance && !IsMultipleApi(api)) {
        throw Error("{} is applied once, without an instance name", Info(api).name);
    }
    const bool multiple = IsMultipleApi(api);
    std::vector<uint32_t> targets;
    for (const uint32_t prim : prims) {
        if (!m_prims.Alive(prim)) {
            throw Error("Prim {} was removed", prim);
        }
        const SchemaId schema = m_prims.At(prim).schema;
        if (!multiple && Includes(schema, api)) {
            throw Error("{} is built into {}, so {} can't lose it", Info(api).name,
                        Info(schema).name, m_paths.Path(prim));
        }
        for (const SchemaId outer : kApiSchemas) {
            if (!multiple && outer != api && m_prims.HasApi(prim, outer) &&
                HasApi(Info(outer).apis, api)) {
                throw Error("{} comes with {}, so {} can't lose it while it has {}", Info(api).name,
                            Info(outer).name, m_paths.Path(prim), Info(outer).name);
            }
        }
        if (multiple) {
            for (const ApiInstance& entry : m_prims.Instances(prim, api)) {
                const std::string_view name = m_tokens.Text(Token{entry.instance});
                if ((!instance || name == *instance) && BuiltinInstance(schema, api, name)) {
                    throw Error("{}:{} is built into {}, so {} can't lose it", Info(api).name, name,
                                Info(schema).name, m_paths.Path(prim));
                }
            }
        }
        if (m_prims.HasApi(prim, api) && std::ranges::find(targets, prim) == targets.end()) {
            targets.push_back(prim);
        }
    }
    if (targets.empty()) {
        return;
    }
    if (m_store.Pinned(api)) {
        throw Error("{} data is mapped or attached, so it can't be removed until the mapping "
                    "closes or the attachment is released",
                    Info(api).name);
    }
    std::set<uint32_t> released;
    for (const uint32_t prim : targets) {
        if (!multiple) {
            released.insert(m_prims.Row(prim, api));
            continue;
        }
        for (const ApiInstance& entry : m_prims.Instances(prim, api)) {
            if (!instance || m_tokens.Text(Token{entry.instance}) == *instance) {
                released.insert(entry.row);
            }
        }
    }
    std::erase_if(m_pending, [&](const PendingTargets& pending) {
        return pending.column.schema == api && released.contains(pending.row);
    });
    if (!multiple) {
        m_prims.Unapply(targets, api);
    } else {
        std::vector<uint32_t> names;
        for (const uint32_t prim : targets) {
            for (const ApiInstance& entry : m_prims.Instances(prim, api)) {
                if (released.contains(entry.row) &&
                    std::ranges::find(names, entry.instance) == names.end()) {
                    names.push_back(entry.instance);
                }
            }
        }
        for (const uint32_t name : names) {
            m_prims.UnapplyInstance(targets, api, name);
        }
    }
    for (const uint32_t prim : targets) {
        Logged(prim, false, api);
    }
    ++m_epoch;
}

void Catalog::RemovePrim(uint32_t prim) {
    ReleaseFiles(prim);
    if (m_paths.Contains(prim)) {
        m_paths.Erase(prim);
    }
    Logged(prim, false);
    m_prims.Remove(prim);
    ++m_epoch;
}

void Catalog::MarkTargets(std::span<const uint32_t> targets) {
    for (const uint32_t target : targets) {
        if (target == kNoPrim) {
            continue;
        }
        if (target >= m_targeted.size()) {
            m_targeted.resize(size_t(target) + 1);
        }
        m_targeted[target] = true;
    }
}

void Catalog::MarkUntrackedTargets() {
    m_untrackedTargets = true;
}

std::vector<Catalog::Referrer> Catalog::Referrers(std::span<const uint32_t> removed) {
    struct Cell {
        uint32_t prim;
        ColumnRef column;
        uint32_t row;
    };

    std::vector<Cell> cells;
    std::vector<device::Download> reads;
    const std::set<uint32_t> gone(removed.begin(), removed.end());
    for (uint32_t prim = 0; prim < m_prims.Size(); ++prim) {
        if (!m_prims.Alive(prim) || gone.contains(prim)) {
            continue;
        }
        const PrimRecord& record = m_prims.At(prim);
        for (const SchemaId table : PrimTables(record.schema, record.apis)) {
            const Span<const ColumnInfo> columns = Columns(table);
            for (uint32_t index = 0; index < columns.size(); ++index) {
                if (columns[index].type != ColumnType::PrimArray) {
                    continue;
                }
                const ColumnRef column{table, &columns[index], index};
                std::vector<uint32_t> rows;
                if (IsMultipleApi(table)) {
                    for (const ApiInstance& entry : m_prims.Instances(prim, table)) {
                        rows.push_back(entry.row);
                    }
                } else {
                    rows.push_back(m_prims.Row(prim, table));
                }
                for (const uint32_t row : rows) {
                    const ArrayRef array = m_store.Array(column, row);
                    if (array.address == 0 || array.count == 0) {
                        continue;
                    }
                    cells.push_back({prim, column, row});
                    reads.push_back(m_store.GetDevice().GetQueue().ReadAsync(
                        m_store.Arrays(ColumnType::PrimArray)
                            .Range(array)
                            .Slice(0, uint64_t(array.count) * sizeof(uint32_t))));
                }
            }
        }
    }
    std::vector<Referrer> referrers;
    for (size_t cell = 0; cell < cells.size(); ++cell) {
        reads[cell].Wait();
        std::vector<uint32_t> targets(reads[cell].Bytes().size() / sizeof(uint32_t));
        std::memcpy(targets.data(), reads[cell].Bytes().data(), targets.size() * sizeof(uint32_t));
        if (std::ranges::any_of(targets, [&](uint32_t target) { return gone.contains(target); })) {
            referrers.push_back(
                {cells[cell].prim, cells[cell].column, cells[cell].row, std::move(targets)});
        }
    }
    return referrers;
}

void Catalog::Remove(uint32_t prim) {
    if (!m_prims.Alive(prim)) {
        throw Error("Prim {} was removed", prim);
    }
    const std::vector<uint32_t> subtree = m_paths.Subtree(prim);
    for (const uint32_t at : subtree) {
        m_prims.CheckRemovable(at);
    }
    const bool targeted = m_untrackedTargets || std::ranges::any_of(subtree, [&](uint32_t at) {
                              return at < m_targeted.size() && m_targeted[at];
                          });

    struct Retarget {
        uint32_t prim;
        uint32_t row;
        ColumnRef column;
        std::vector<std::string> paths;
    };

    std::vector<Retarget> retargets;
    if (targeted) {
        m_store.Flush();
        for (const Referrer& referrer : Referrers(subtree)) {
            const auto pending = std::ranges::find_if(m_pending, [&](const PendingTargets& entry) {
                return entry.row == referrer.row && entry.column.schema == referrer.column.schema &&
                       entry.column.index == referrer.column.index;
            });
            std::vector<std::string> paths;
            if (pending != m_pending.end()) {
                paths = pending->paths;
            } else {
                for (const uint32_t target : referrer.targets) {
                    if (target != kNoPrim && m_prims.Alive(target)) {
                        paths.push_back(m_paths.Path(target));
                    }
                }
            }
            retargets.push_back({referrer.prim, referrer.row, referrer.column, std::move(paths)});
        }
    }
    for (const uint32_t at : std::views::reverse(subtree)) {
        RemovePrim(at);
        if (at < m_targeted.size()) {
            m_targeted[at] = false;
        }
    }
    for (const Retarget& retarget : retargets) {
        SetTargets(retarget.prim, retarget.row, retarget.column, retarget.paths);
        m_store.StampWorld(m_prims.At(retarget.prim).world);
    }
}

void Catalog::Move(uint32_t prim, std::string_view path) {
    if (!m_prims.Alive(prim)) {
        throw Error("Prim {} was removed", prim);
    }
    const uint32_t parent = m_paths.FindParent(path);
    if (parent == Paths::kNone && path.rfind('/') != 0) {
        throw Error("Path \"{}\" has no parent", path);
    }
    const PrimRecord& record = m_prims.At(prim);
    const uint32_t world = parent == Paths::kRoot ? kGlobalWorld : m_prims.At(parent).world;
    const bool isWorld = IsA(record.schema, SchemaId::World);
    if (isWorld ? world != kGlobalWorld : world != record.world) {
        throw Error("\"{}\" can't move to \"{}\": prims stay in their world, and worlds don't nest",
                    m_paths.Path(prim), path);
    }
    const std::vector<uint32_t> before = m_paths.Subtree(prim);
    m_paths.Move(prim, path);
    for (const uint32_t at : before) {
        Logged(at, false);
        m_prims.Renew(at);
    }
    m_prims.Reparent(prim, parent);
    std::vector<std::string> moved;
    for (const uint32_t at : before) {
        Logged(at, true);
        moved.push_back(m_paths.Path(at));
    }
    ResolvePending(moved);
    ++m_epoch;
}

void Catalog::SetTargets(uint32_t prim, uint32_t row, ColumnRef column,
                         std::span<const std::string> paths) {
    std::erase_if(m_pending, [&](const PendingTargets& pending) {
        return pending.row == row && pending.column.schema == column.schema &&
               pending.column.index == column.index;
    });
    const std::vector<uint32_t> targets = ResolveTargets(paths);
    MarkTargets(targets);
    ArrayHeap& arrays = m_store.Arrays(column.column->type);
    const ArrayRef array = arrays.Allocate(uint32_t(targets.size()));
    arrays.Write(array, std::as_bytes(std::span(targets)));
    m_store.SetArray(column, row, array);
    if (std::ranges::find(targets, kNoPrim) != targets.end()) {
        m_pending.push_back({prim, m_prims.Generation(prim), column, row,
                             std::vector<std::string>(paths.begin(), paths.end())});
    }
}

std::vector<uint32_t> Catalog::ResolveTargets(std::span<const std::string> paths) const {
    std::vector<uint32_t> targets;
    targets.reserve(paths.size());
    for (const std::string& path : paths) {
        targets.push_back(m_paths.Find(path));
    }
    return targets;
}

void Catalog::ResolvePending(std::span<const std::string> defined) {
    std::erase_if(m_pending, [&](const PendingTargets& pending) {
        if (!m_prims.Alive(pending.prim) ||
            m_prims.Generation(pending.prim) != pending.generation) {
            return true;
        }
        const bool named = std::ranges::any_of(defined, [&](const std::string& path) {
            return std::ranges::find(pending.paths, path) != pending.paths.end();
        });
        if (!named) {
            return false;
        }
        const std::vector<uint32_t> targets = ResolveTargets(pending.paths);
        MarkTargets(targets);
        ArrayHeap& arrays = m_store.Arrays(pending.column.column->type);
        const ArrayRef array = arrays.Allocate(uint32_t(targets.size()));
        arrays.Write(array, std::as_bytes(std::span(targets)));
        m_store.SetArray(pending.column, pending.row, array);
        return std::ranges::find(targets, kNoPrim) == targets.end();
    });
}

PrimBlock Catalog::Clone(uint32_t source, std::span<const std::string> targets,
                         const CloneDesc& desc) {
    if (!m_prims.Alive(source)) {
        throw Error("Prim {} was removed", source);
    }
    const std::vector<uint32_t> sources = m_paths.Subtree(source);
    const std::string root = m_paths.Path(source);
    std::vector<uint32_t> parents;
    std::vector<std::string> paths;
    if (desc.moveSource) {
        for (const uint32_t prim : sources) {
            m_prims.CheckRemovable(prim);
        }
        parents.push_back(m_prims.At(source).parent);
        paths.push_back(root);
    }
    std::set<std::string_view> seen;
    for (const std::string& target : targets) {
        if (!seen.insert(target).second) {
            throw Error("\"{}\" is a target twice", target);
        }
        if (target.starts_with(root + "/")) {
            throw Error("\"{}\" is inside the clone's source \"{}\"", target, root);
        }
        parents.push_back(NewParent(target));
        paths.push_back(target);
    }

    BlockShape shape;
    std::unordered_map<uint32_t, uint32_t> position;
    std::vector<uint32_t> worlds;
    for (uint32_t i = 0; i < sources.size(); ++i) {
        const PrimRecord& record = m_prims.At(sources[i]);
        position[sources[i]] = i;
        shape.schemas.push_back(record.schema);
        shape.parents.push_back(i == 0 ? kNoPrim : position.at(record.parent));
        shape.apis.push_back(record.apis);
        if (IsA(record.schema, SchemaId::World)) {
            worlds.push_back(record.world);
        }
    }
    bool reusable = desc.moveSource && !worlds.empty() && worlds.back() + 1 == m_prims.WorldCount();
    for (uint32_t i = 1; i < worlds.size(); ++i) {
        reusable = reusable && worlds[i] == worlds[i - 1] + 1;
    }

    PathShape pathShape = m_paths.Shape(source);
    const PrimBlock block =
        m_prims.Add(shape, parents, reusable ? worlds.front() : kGlobalWorld, desc.attachable);
    for (const BlockTable& table : block.tables) {
        std::vector<uint32_t> rows;
        for (const uint32_t prim : sources) {
            if (const uint32_t row = m_prims.FindRow(prim, table.table); row != kNoRow) {
                rows.push_back(row);
            }
        }
        m_store.CloneRows(table.table, rows, table.first, block.copies);
    }
    CloneArrays(block, sources);

    std::vector<std::pair<uint32_t, std::string>> owned;
    for (uint32_t index = 0; index < sources.size(); ++index) {
        const auto [first, last] = m_fileOwners.equal_range(sources[index]);
        for (auto owner = first; owner != last; ++owner) {
            for (uint32_t copy = 0; copy < block.copies; ++copy) {
                owned.emplace_back(block.Prim(copy, index), owner->second);
            }
        }
    }
    for (const auto& [prim, path] : owned) {
        OwnFile(prim, path);
    }
    if (desc.moveSource) {
        for (const uint32_t prim : std::views::reverse(sources)) {
            RemovePrim(prim);
        }
    }
    m_paths.InsertClones(std::move(pathShape), paths, block.firstPrim);
    for (uint32_t prim = block.firstPrim; prim < block.firstPrim + block.PrimCount(); ++prim) {
        Logged(prim, true);
    }
    ++m_epoch;
    return block;
}

void Catalog::CloneArrays(const PrimBlock& block, std::span<const uint32_t> sources) {
    m_store.Flush();
    std::unordered_map<uint32_t, uint32_t> position;
    for (uint32_t i = 0; i < sources.size(); ++i) {
        position[sources[i]] = i;
    }
    for (const BlockTable& table : block.tables) {
        std::vector<uint32_t> rows;
        for (const uint32_t prim : sources) {
            if (const uint32_t row = m_prims.FindRow(prim, table.table); row != kNoRow) {
                rows.push_back(row);
            }
        }
        CloneTableArrays(table.table, rows, table.first, table.perCopy, block, position);
    }
    CloneInstances(block, sources, position);
}

void Catalog::CloneInstances(const PrimBlock& block, std::span<const uint32_t> sources,
                             const std::unordered_map<uint32_t, uint32_t>& position) {
    for (uint32_t i = 0; i < sources.size(); ++i) {
        const uint64_t apis = m_prims.At(sources[i]).apis;
        std::vector<uint32_t> copies;
        for (uint32_t copy = 0; copy < block.copies; ++copy) {
            copies.push_back(block.Prim(copy, i));
        }
        for (const SchemaId api : kApiSchemas) {
            if (!IsMultipleApi(api) || !HasApi(apis, api)) {
                continue;
            }
            const std::span<const ApiInstance> listed = m_prims.Instances(sources[i], api);
            const std::vector<ApiInstance> instances(listed.begin(), listed.end());
            for (const ApiInstance& entry : std::views::reverse(instances)) {
                const BlockTable rows = m_prims.ApplyInstance(copies, api, entry.instance);
                const uint32_t source = entry.row;
                m_store.CloneRows(api, std::span(&source, 1), rows.first, block.copies);
                m_prims.StageLinks(copies, api);
                CloneTableArrays(api, std::span(&source, 1), rows.first, 1, block, position);
            }
        }
    }
}

void Catalog::CloneTableArrays(SchemaId table, std::span<const uint32_t> rows, uint32_t first,
                               uint32_t perCopy, const PrimBlock& block,
                               const std::unordered_map<uint32_t, uint32_t>& position) {
    for (uint32_t index = 0; index < Info(table).columnCount; ++index) {
        const ColumnRef column{table, &Columns(table)[index], index};
        if (!IsArray(column.column->type)) {
            continue;
        }
        ArrayHeap& arrays = m_store.Arrays(column.column->type);
        for (uint32_t j = 0; j < rows.size(); ++j) {
            const ArrayRef array = m_store.Array(column, rows[j]);
            if (array.address == 0) {
                continue;
            }
            const bool exclusive = m_store.Exclusive(column, rows[j]);
            std::vector<uint32_t> targets;
            if (column.column->type == ColumnType::PrimArray) {
                targets = Items<uint32_t>(arrays.Read(array));
            }
            const bool inside = std::ranges::any_of(
                targets, [&](uint32_t target) { return position.contains(target); });
            for (uint32_t copy = 0; copy < block.copies; ++copy) {
                const uint32_t row = first + copy * perCopy + j;
                if (inside) {
                    std::vector<uint32_t> rebased = targets;
                    for (uint32_t& target : rebased) {
                        if (const auto at = position.find(target); at != position.end()) {
                            target = block.Prim(copy, at->second);
                        }
                    }
                    MarkTargets(rebased);
                    const ArrayRef owned = arrays.Allocate(uint32_t(rebased.size()));
                    arrays.Write(owned, std::as_bytes(std::span(rebased)));
                    m_store.SetArray(column, row, owned);
                } else if (exclusive) {
                    const ArrayRef owned = arrays.Allocate(array.count);
                    const uint64_t bytes = uint64_t(array.count) * arrays.ItemBytes();
                    device::Queue& queue = m_store.GetDevice().GetQueue();
                    device::CommandEncoder encoder = queue.CreateEncoder();
                    encoder.Copy(arrays.Range(array).Slice(0, bytes),
                                 arrays.Range(owned).Slice(0, bytes));
                    queue.Submit(encoder.Finish());
                    m_store.SetArray(column, row, owned);
                } else {
                    m_store.SetArray(column, row, arrays.Share(array));
                }
            }
        }
    }
}

void Catalog::AddFile(std::string path, FileBytes bytes, std::optional<uint32_t> owner) {
    if (!bytes) {
        throw Error("Adding file \"{}\": it has no bytes", path);
    }
    if (owner && !m_prims.Alive(*owner)) {
        throw Error("Prim {} was removed", *owner);
    }
    m_files[path].bytes = std::move(bytes);
    if (owner) {
        OwnFile(*owner, path);
    }
}

void Catalog::RemoveFile(std::string_view path) {
    const auto found = m_files.find(path);
    if (found == m_files.end()) {
        return;
    }
    m_files.erase(found);
    std::erase_if(m_fileOwners, [&](const auto& owner) { return owner.second == path; });
}

FileBytes Catalog::FindFile(std::string_view path) const {
    const auto found = m_files.find(path);
    return found == m_files.end() ? nullptr : found->second.bytes;
}

std::vector<std::pair<std::string, FileBytes>> Catalog::OwnedFiles(uint32_t prim) const {
    std::vector<std::pair<std::string, FileBytes>> files;
    const auto [first, last] = m_fileOwners.equal_range(prim);
    for (auto owner = first; owner != last; ++owner) {
        if (const auto file = m_files.find(owner->second); file != m_files.end()) {
            files.emplace_back(owner->second, file->second.bytes);
        }
    }
    return files;
}

void Catalog::OwnFile(uint32_t prim, const std::string& path) {
    ++m_files.at(path).owners;
    m_fileOwners.emplace(prim, path);
}

void Catalog::ReleaseFiles(uint32_t prim) {
    const auto [first, last] = m_fileOwners.equal_range(prim);
    for (auto owner = first; owner != last; ++owner) {
        const auto file = m_files.find(owner->second);
        if (file != m_files.end() && --file->second.owners == 0) {
            m_files.erase(file);
        }
    }
    m_fileOwners.erase(first, last);
}

}

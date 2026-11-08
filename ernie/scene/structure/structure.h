// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/schema/schema.h"
#include "ernie/scene/primvar.h"
#include "ernie/scene/structure/paths.h"
#include "ernie/scene/structure/prims.h"
#include "ernie/scene/structure/tokens.h"
#include "ernie/scene/types.h"

namespace ernie {

class Structure;

struct StructureEntry {
    Ordinal ordinal = 0;
    uint32_t prim = kNoPrim;
    uint32_t generation = 0;
    uint32_t world = kGlobalWorld;
    bool added = false;
    SchemaId api = SchemaId::None;
};

class RowWriter {
  public:
    RowWriter(Structure& structure, const PrimBlock& block, std::span<const uint32_t> prims = {})
        : m_structure(structure), m_block(block), m_prims(prims) {}

    template <class T> void Set(SchemaId owner, uint32_t column, const T& value) {
        const ColumnRef ref{owner, &Columns(owner)[column], column};
        if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            if (ref.column->type == ColumnType::TokenArray) {
                SetTokens(ref, value);
            } else {
                SetTargets(ref, value);
            }
        } else if constexpr (std::is_same_v<T, std::vector<Primvar>>) {
            SetPrimvars(ref, value);
        } else if constexpr (std::is_same_v<T, bool>) {
            const uint32_t word = value ? 1 : 0;
            Fill(ref, std::as_bytes(std::span(&word, 1)));
        } else if constexpr (std::is_convertible_v<const T&, std::string_view>) {
            SetToken(ref, value);
        } else if constexpr (requires {
                                 value.data();
                                 value.size();
                             }) {
            SetArray(ref, std::as_bytes(std::span(value.data(), value.size())),
                     uint32_t(value.size()));
        } else {
            static_assert(std::is_trivially_copyable_v<T>, "cells are copied as bytes");
            Fill(ref, std::as_bytes(std::span(&value, 1)));
        }
    }

    template <class F> void Instance(SchemaId api, std::string_view instance, const F& write);

  private:
    std::vector<uint32_t> Prims() const;
    const BlockTable& Rows(SchemaId table) const;
    uint32_t RowCount(const BlockTable& rows) const;
    void Fill(ColumnRef column, std::span<const std::byte> cell);
    void SetToken(ColumnRef column, std::string_view text);
    void SetArray(ColumnRef column, std::span<const std::byte> bytes, uint32_t count);
    void SetTokens(ColumnRef column, std::span<const std::string> texts);
    void SetTargets(ColumnRef column, std::span<const std::string> paths);
    void SetPrimvars(ColumnRef column, std::span<const Primvar> primvars);

    Structure& m_structure;
    const PrimBlock& m_block;
    std::span<const uint32_t> m_prims;
};

class Structure {
  public:
    explicit Structure(ColumnStore& store);

    ColumnStore& GetColumnStore() {
        return m_store;
    }

    Tokens& GetTokens() {
        return m_tokens;
    }

    const Tokens& GetTokens() const {
        return m_tokens;
    }

    const Paths& GetPaths() const {
        return m_paths;
    }

    const Prims& GetPrims() const {
        return m_prims;
    }

    void MarkSession(uint32_t prim) {
        m_prims.MarkSession(prim);
    }

    uint64_t Epoch() const {
        return m_epoch;
    }

    const std::map<uint32_t, ComposedWorldDesc>& Compositions() const {
        return m_compositions;
    }

    uint64_t CompositionVersion() const {
        return m_compositionVersion;
    }

    bool Compose(uint32_t world, ComposedWorldDesc desc) {
        if (const auto found = m_compositions.find(world);
            found != m_compositions.end() && found->second.worlds == desc.worlds &&
            found->second.offsets == desc.offsets && found->second.global == desc.global &&
            found->second.stageLights == desc.stageLights) {
            return false;
        }
        m_compositions[world] = std::move(desc);
        ++m_compositionVersion;
        ++m_epoch;
        return true;
    }

    void Decompose(uint32_t world) {
        if (m_compositions.erase(world) > 0) {
            ++m_compositionVersion;
            ++m_epoch;
        }
    }

    std::span<const StructureEntry> Log() const {
        return m_log;
    }

    uint32_t Find(std::string_view path) const;
    std::vector<uint32_t> Select(std::string_view pattern, SchemaId schema) const;

    PrimBlock Define(SchemaId schema, std::span<const std::string> paths);

    template <class T>
    PrimBlock Define(std::span<const std::string> paths, const typename T::Desc& desc) {
        return Define(T::kSchema, paths, [&](RowWriter& writer) { T::Write(writer, desc); });
    }

    template <class Write>
    PrimBlock Define(SchemaId schema, std::span<const std::string> paths, const Write& write,
                     bool attachable = false) {
        return Define(schema, paths, 0, write, attachable);
    }

    template <class Write>
    PrimBlock Define(SchemaId schema, std::span<const std::string> paths, uint64_t apis,
                     const Write& write, bool attachable = false) {
        PrimBlock block = Allocate(schema, paths, apis, attachable);
        try {
            RowWriter writer(*this, block);
            write(writer);
        } catch (...) {
            RemoveBlock(block);
            throw;
        }
        ResolvePending(paths);
        return block;
    }

    PrimBlock Clone(uint32_t source, std::span<const std::string> targets,
                    const CloneDesc& desc = {});
    void SetTargets(uint32_t prim, uint32_t row, ColumnRef column,
                    std::span<const std::string> paths);
    void Remove(uint32_t prim);
    void Move(uint32_t prim, std::string_view path);

    template <class Write>
    void Apply(std::span<const uint32_t> prims, SchemaId api, std::string_view instance,
               const Write& write) {
        std::vector<uint32_t> targets;
        for (const uint32_t prim : Applicable(prims, api, true)) {
            if (m_prims.InstanceRow(prim, api, m_tokens.Intern(instance).id) == kNoRow) {
                targets.push_back(prim);
            }
        }
        if (targets.empty()) {
            return;
        }
        ApplyInstance(targets, api, instance, write);
        for (const uint32_t prim : targets) {
            Logged(prim, true, api);
        }
        ++m_epoch;
    }

    template <class Write>
    void ApplyInstance(std::span<const uint32_t> prims, SchemaId api, std::string_view instance,
                       const Write& write) {
        const uint32_t token = m_tokens.Intern(instance).id;
        PrimBlock block;
        block.count = 1;
        block.copies = uint32_t(prims.size());
        block.tables.push_back(m_prims.ApplyInstance(prims, api, token));
        try {
            RowWriter writer(*this, block, prims);
            write(writer);
        } catch (...) {
            m_prims.UnapplyInstance(prims, api, token);
            throw;
        }
    }

    template <class Write>
    void Apply(std::span<const uint32_t> prims, SchemaId api, const Write& write) {
        const std::vector<uint32_t> targets = Applicable(prims, api, false);
        if (targets.empty()) {
            return;
        }
        PrimBlock block;
        block.count = 1;
        block.copies = uint32_t(targets.size());
        std::vector<std::pair<SchemaId, std::vector<uint32_t>>> applied;
        for (const SchemaId inner : kApiSchemas) {
            if (IsMultipleApi(inner) || !HasApi(Info(api).apis, inner)) {
                continue;
            }
            std::vector<uint32_t> lacking;
            for (const uint32_t prim : targets) {
                if (!m_prims.HasApi(prim, inner)) {
                    lacking.push_back(prim);
                }
            }
            BlockTable rows{inner, 0, 1, 0};
            if (!lacking.empty()) {
                rows = m_prims.Apply(lacking, inner);
                rows.count = uint32_t(lacking.size());
            }
            block.tables.push_back(rows);
            applied.emplace_back(inner, std::move(lacking));
        }
        block.tables.push_back(m_prims.Apply(targets, api));
        applied.emplace_back(api, targets);
        try {
            RowWriter writer(*this, block, targets);
            write(writer);
        } catch (...) {
            for (const auto& [table, members] : applied) {
                m_prims.Unapply(members, table);
            }
            throw;
        }
        for (const auto& [table, members] : applied) {
            for (const uint32_t prim : members) {
                Logged(prim, true, table);
            }
        }
        ++m_epoch;
    }

    void Apply(std::span<const uint32_t> prims, SchemaId api);
    void Unapply(std::span<const uint32_t> prims, SchemaId api,
                 std::optional<std::string_view> instance = std::nullopt);

    void MarkTargets(std::span<const uint32_t> targets);
    void MarkUntrackedTargets();

    void AddFile(std::string path, FileBytes bytes, std::optional<uint32_t> owner);
    std::vector<std::pair<std::string, FileBytes>> OwnedFiles(uint32_t prim) const;
    void RemoveFile(std::string_view path);
    FileBytes FindFile(std::string_view path) const;

  private:
    friend class RowWriter;

    struct PendingTargets {
        uint32_t prim = kNoPrim;
        uint32_t generation = 0;
        ColumnRef column;
        uint32_t row = kNoRow;
        std::vector<std::string> paths;
    };

    struct MemoryFile {
        FileBytes bytes;
        uint32_t owners = 0;
    };

    uint32_t NewParent(std::string_view path) const;
    PrimBlock Allocate(SchemaId schema, std::span<const std::string> paths, uint64_t apis,
                       bool attachable);
    std::vector<uint32_t> Applicable(std::span<const uint32_t> prims, SchemaId api,
                                     bool multiple) const;
    void CloneInstances(const PrimBlock& block, std::span<const uint32_t> sources,
                        const std::unordered_map<uint32_t, uint32_t>& position);
    void CloneTableArrays(SchemaId table, std::span<const uint32_t> rows, uint32_t first,
                          uint32_t perCopy, const PrimBlock& block,
                          const std::unordered_map<uint32_t, uint32_t>& position);
    void RemoveBlock(const PrimBlock& block);
    void RemovePrim(uint32_t prim);
    void Logged(uint32_t prim, bool added, SchemaId api = SchemaId::None);
    std::vector<uint32_t> ResolveTargets(std::span<const std::string> paths) const;
    void ResolvePending(std::span<const std::string> defined);

    struct Referrer {
        uint32_t prim = kNoPrim;
        ColumnRef column;
        uint32_t row = kNoRow;
        std::vector<uint32_t> targets;
    };

    std::vector<Referrer> Referrers(std::span<const uint32_t> removed);
    void CloneArrays(const PrimBlock& block, std::span<const uint32_t> sources);
    void OwnFile(uint32_t prim, const std::string& path);
    void ReleaseFiles(uint32_t prim);

    ColumnStore& m_store;
    Tokens m_tokens;
    Paths m_paths;
    Prims m_prims;
    uint64_t m_epoch = 0;
    std::map<uint32_t, ComposedWorldDesc> m_compositions;
    uint64_t m_compositionVersion = 0;
    std::vector<StructureEntry> m_log;
    std::vector<PendingTargets> m_pending;
    std::vector<bool> m_targeted;
    bool m_untrackedTargets = false;
    std::map<std::string, MemoryFile, std::less<>> m_files;
    std::unordered_multimap<uint32_t, std::string> m_fileOwners;
};

template <class F>
void RowWriter::Instance(SchemaId api, std::string_view instance, const F& write) {
    m_structure.ApplyInstance(Prims(), api, instance, [&](RowWriter& writer) { write(writer); });
}

}

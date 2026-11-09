// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "ernie/scene/records/render_tables.h"
#include "ernie/stage/publish/header.h"
#include "ernie/stage/publish/table.h"

namespace ernie {

enum class TableId : uint32_t {
    Transforms,
    Imageables,
    Meshes,
    MeshPoints,
    Points,
    PointsMoved,
    Tangents,
    Textures,
    Instances,
    Slots,
};

class RenderTable : public Table {
  public:
    virtual void Describe(RenderTables&) const {}
};

class RenderHeader : public SnapshotHeader {
  public:
    const RenderTables& Tables() const {
        return m_tables;
    }

    void Describe(const Table& table) override {
        if (const auto* render = dynamic_cast<const RenderTable*>(&table)) {
            render->Describe(m_tables);
        }
    }

    std::span<const std::byte> Seal(const AuthoredData& data, uint32_t worldCount) override {
        m_tables.prims = {.schema = data.prims.schema,
                          .world = data.prims.world,
                          .flags = data.prims.flags,
                          .slots = data.prims.slots,
                          .apis = data.prims.apis,
                          .rows = data.rows,
                          .parent = data.prims.parent,
                          .count = data.primCount};
        m_tables.worldCount = worldCount;
        return std::as_bytes(std::span(&m_tables, 1));
    }

  private:
    RenderTables m_tables{};
};

}

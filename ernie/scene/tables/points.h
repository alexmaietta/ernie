// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ernie/bvh/builder.h"
#include "ernie/device/buffer.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/records/points.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"

namespace ernie {

struct PointsStats {
    uint32_t clouds = 0;
    uint32_t built = 0;
    uint32_t points = 0;
};

class PointsTable : public Table {
  public:
    PointsTable(Structure& structure, bvh::Builder& builder)
        : m_structure(structure), m_builder(builder) {}

    std::string_view Name() const override {
        return "points";
    }

    std::span<const ColumnRef> Inputs() const override;
    std::span<const TableId> TableOutputs() const override;
    void Describe(RenderTables& tables) const override;
    void Prepare() override;
    void Run(const ChangeSet& changes, TableVersions& versions) override;
    void Hold(std::vector<Ref<const RefCounted>>& held) const override;

    const PointsStats& Stats() const {
        return m_stats;
    }

    const PointsRecord& Record(uint32_t pointsRow) const {
        return m_records.Row(pointsRow);
    }

  private:
    struct Cloud {
        uint32_t prim = kNoPrim;
        uint32_t generation = 0;
        uint32_t pointBasedRow = kNoRow;
        uint32_t row = kNoRow;
        uint32_t capacity = 0;
        Ref<bvh::Blas> blas;
        Ref<device::Buffer> boxes;
        ArrayRef bounds{};
        bool build = false;
    };

    void Place(uint32_t prim);
    void Forget(uint32_t prim);
    void Link(Cloud& cloud);
    void Build(std::vector<uint32_t>& updated);
    device::Program& Kernels();

    Structure& m_structure;
    bvh::Builder& m_builder;
    Ref<device::Program> m_program;
    std::unordered_map<uint32_t, Cloud> m_clouds;
    std::unordered_map<uint32_t, uint32_t> m_byPointBasedRow;
    std::unordered_map<uint32_t, uint32_t> m_byRow;
    RecordTable<PointsRecord> m_records{"points records"};
    std::vector<uint32_t> m_relinked;
    PointsStats m_stats;
};

}

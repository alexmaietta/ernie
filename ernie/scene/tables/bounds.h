// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "ernie/bvh/kernels/bvh.h"
#include "ernie/device/buffer.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/records/bounds.h"
#include "ernie/scene/tables/instances.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"
#include "ernie/stage/structure/structure.h"

namespace ernie {

class BoundsTable : public RenderTable {
  public:
    BoundsTable(Structure& structure, const RenderTables& tables, const InstanceTable& instances)
        : m_structure(structure), m_tables(tables), m_instances(instances) {}

    std::string_view Name() const override {
        return "bounds";
    }

    std::span<const ColumnRef> Inputs() const override;

    std::span<const TableId> TableInputs() const override {
        static constexpr TableId kTables[] = {
            TableId::Transforms, TableId::Imageables,  TableId::Meshes,   TableId::MeshPoints,
            TableId::Points,     TableId::PointsMoved, TableId::Instances};
        return kTables;
    }

    void Prepare() override;
    void Run(const ChangeSet& changes, TableVersions& versions) override;

    void Describe(RenderTables& tables) const override;

    const Ref<device::Buffer>& PrimBounds() const {
        return m_primBounds.Buffer();
    }

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_boxes.Buffer(), m_primBounds.Buffer()});
    }

    std::optional<BvhBox> World(uint32_t world) const;
    device::Download ReadWorldAsync(uint32_t world) const;
    device::Download ReadWorldsAsync() const;

  private:
    void Relayout();

    Structure& m_structure;
    const RenderTables& m_tables;
    const InstanceTable& m_instances;
    device::Program& Kernels();

    Ref<device::Program> m_program;
    Ref<device::Buffer> m_origins;
    Ref<device::Buffer> m_extents;
    DeviceTable<BvhBox> m_boxes{"world bounds"};
    DeviceTable<BoundableBox> m_primBounds{"prim bounds"};
    Ref<device::Buffer> m_instancerExtents;
    uint64_t m_epoch = ~uint64_t(0);
};

}

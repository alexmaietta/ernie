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
#include "ernie/scene/catalog/catalog.h"
#include "ernie/scene/systems/instances.h"
#include "ernie/scene/systems/system.h"

namespace ernie {

class Bounds : public System {
  public:
    Bounds(Catalog& catalog, const Instances& instances)
        : m_catalog(catalog), m_instances(instances) {}

    std::string_view Name() const override {
        return "bounds";
    }

    std::span<const ColumnRef> Inputs() const override;

    std::span<const TableId> TableInputs() const override {
        static constexpr TableId kTables[] = {TableId::Transforms, TableId::Imageables,
                                              TableId::Meshes, TableId::MeshPoints};
        return kTables;
    }

    void Prepare() override;
    void Run(const ChangeSet& changes, TableLog& log) override;

    const Ref<device::Buffer>& Boxes() const {
        return m_boxes;
    }

    void Describe(RenderTables& tables) const override;

    const Ref<device::Buffer>& Boundables() const {
        return m_boundables;
    }

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_boxes, m_boundables});
    }

    std::optional<BvhBox> World(uint32_t world) const;
    device::Download ReadWorldAsync(uint32_t world) const;
    device::Download ReadWorldsAsync() const;

  private:
    void Relayout();

    Catalog& m_catalog;
    const Instances& m_instances;
    device::Program& Kernels();

    Ref<device::Program> m_program;
    Ref<device::Buffer> m_origins;
    Ref<device::Buffer> m_extents;
    Ref<device::Buffer> m_boxes;
    Ref<device::Buffer> m_boundables;
    uint32_t m_boundableCapacity = 0;
    Ref<device::Buffer> m_instancerExtents;
    uint64_t m_epoch = ~uint64_t(0);
    uint32_t m_slots = 0;
};

}

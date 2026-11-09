// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/records/instances.h"
#include "ernie/scene/tables/instances.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"
#include "ernie/stage/structure/structure.h"

namespace ernie {

class SurfaceTable : public RenderTable {
  public:
    SurfaceTable(Structure& structure, const RenderTables& tables, const InstanceTable& instances)
        : m_structure(structure), m_tables(tables), m_instances(instances) {}

    std::string_view Name() const override {
        return "surfaces";
    }

    std::span<const ColumnRef> Inputs() const override;

    std::span<const TableId> TableInputs() const override {
        static constexpr TableId kTables[] = {TableId::Meshes, TableId::Slots};
        return kTables;
    }

    bool Structural() const override {
        return false;
    }

    void Prepare() override;
    void Run(const ChangeSet& changes, TableVersions& versions) override;
    void Describe(RenderTables& tables) const override;

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_surfaces.Buffer()});
    }

    uint32_t Written() const {
        return m_written;
    }

  private:
    device::Program& Kernels();

    Structure& m_structure;
    const RenderTables& m_tables;
    const InstanceTable& m_instances;
    Ref<device::Program> m_program;
    DeviceTable<SurfaceRecord> m_surfaces{"instance surfaces"};
    uint32_t m_written = 0;
};

}

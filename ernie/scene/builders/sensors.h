// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/builders/builder.h"
#include "ernie/scene/builders/table_storage.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/sensors.h"

namespace ernie {

class SensorBuilder : public TableBuilder {
  public:
    explicit SensorBuilder(Structure& structure) : m_structure(structure) {}

    std::string_view Name() const override {
        return "sensors";
    }

    std::span<const ColumnRef> Inputs() const override;
    void Prepare() override;
    void Run(const ChangeSet& changes, TableWrites& writes) override;
    void Describe(RenderTables& tables) const override;

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_cameras.Buffer(), m_lidars.Buffer(), m_radars.Buffer()});
    }

  private:
    device::Program& Kernels();

    Structure& m_structure;
    Ref<device::Program> m_program;
    DeviceTable<CameraRecord> m_cameras{"camera records"};
    DeviceTable<LidarRecord> m_lidars{"lidar records"};
    DeviceTable<RadarRecord> m_radars{"radar records"};
};

}

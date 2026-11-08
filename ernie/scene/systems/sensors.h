// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/catalog/catalog.h"
#include "ernie/scene/kernels/tables/sensors.h"
#include "ernie/scene/systems/system.h"
#include "ernie/scene/systems/tables.h"

namespace ernie {

class Sensors : public System {
  public:
    explicit Sensors(Catalog& catalog) : m_catalog(catalog) {}

    std::string_view Name() const override {
        return "sensors";
    }

    std::span<const ColumnRef> Inputs() const override;
    void Prepare() override;
    void Run(const ChangeSet& changes, TableLog& log) override;
    void Describe(RenderTables& tables) const override;

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_cameras.Buffer(), m_lidars.Buffer(), m_radars.Buffer()});
    }

  private:
    device::Program& Kernels();

    Catalog& m_catalog;
    Ref<device::Program> m_program;
    DeviceTable<CameraRecord> m_cameras{"camera records"};
    DeviceTable<LidarRecord> m_lidars{"lidar records"};
    DeviceTable<RadarRecord> m_radars{"radar records"};
};

}

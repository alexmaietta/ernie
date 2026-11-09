// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/records/sensors.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"
#include "ernie/stage/structure/structure.h"

namespace ernie {

class SensorTable : public RenderTable {
  public:
    explicit SensorTable(Structure& structure) : m_structure(structure) {}

    std::string_view Name() const override {
        return "sensors";
    }

    std::span<const ColumnRef> Inputs() const override;
    void Prepare() override;
    void Run(const ChangeSet& changes, TableVersions& versions) override;
    void Describe(RenderTables& tables) const override;

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_cameras.Buffer(), m_distortions.Buffer(), m_detectors.Buffer(),
                           m_lidars.Buffer(), m_radars.Buffer()});
    }

  private:
    device::Program& Kernels();

    Structure& m_structure;
    Ref<device::Program> m_program;
    DeviceTable<CameraRecord> m_cameras{"camera records"};
    DeviceTable<DistortionRecord> m_distortions{"distortion records"};
    DeviceTable<DetectorRecord> m_detectors{"detector records"};
    DeviceTable<LidarRecord> m_lidars{"lidar records"};
    DeviceTable<RadarRecord> m_radars{"radar records"};
};

}

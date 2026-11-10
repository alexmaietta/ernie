// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/sensors.h"

#include <algorithm>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/kernels/sensors.h"
#include "ernie/stage/publish/change_set.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

std::vector<ColumnRef> SensorInputs() {
    std::vector<ColumnRef> inputs;
    for (const SchemaId schema : {SchemaId::Camera, SchemaId::LensDistortionAPI,
                                  SchemaId::DetectorAPI, SchemaId::Lidar, SchemaId::Radar}) {
        const Span<const ColumnInfo> columns = Columns(schema);
        for (uint32_t index = 0; index < columns.size(); ++index) {
            inputs.push_back({schema, &columns[index], index});
        }
    }
    return inputs;
}

}

std::span<const ColumnRef> SensorTable::Inputs() const {
    static const std::vector<ColumnRef> inputs = SensorInputs();
    return inputs;
}

void SensorTable::Describe(RenderTables& tables) const {
    tables.cameras.address = m_cameras.Address();
    tables.distortions.address = m_distortions.Address();
    tables.detectors.address = m_detectors.Address();
    tables.lidars.address = m_lidars.Address();
    tables.radars.address = m_radars.Address();
}

void SensorTable::Run(const ChangeSet&, TableVersions& versions) {
    ERNIE_ZONE("Sensors/Run");
    versions.Record(TableId::Cameras, std::vector<uint32_t>{kGlobalWorld});
    ColumnStore& store = m_structure.GetColumnStore();
    device::Device& device = store.GetDevice();
    SensorParams params{};
    params.cameraRows = store.Schema(SchemaId::Camera).Rows();
    params.distortionRows = store.Schema(SchemaId::LensDistortionAPI).Rows();
    params.detectorRows = store.Schema(SchemaId::DetectorAPI).Rows();
    params.lidarRows = store.Schema(SchemaId::Lidar).Rows();
    params.radarRows = store.Schema(SchemaId::Radar).Rows();
    m_cameras.Reserve(device, store.Schema(SchemaId::Camera).Capacity());
    m_distortions.Reserve(device, store.Schema(SchemaId::LensDistortionAPI).Capacity());
    m_detectors.Reserve(device, store.Schema(SchemaId::DetectorAPI).Capacity());
    m_lidars.Reserve(device, store.Schema(SchemaId::Lidar).Capacity());
    m_radars.Reserve(device, store.Schema(SchemaId::Radar).Capacity());
    params.scene.address = store.DeviceData();
    params.cameras.address = m_cameras.Address();
    params.distortions.address = m_distortions.Address();
    params.detectors.address = m_detectors.Address();
    params.lidars.address = m_lidars.Address();
    params.radars.address = m_radars.Address();
    device::Queue& queue = device.GetQueue();
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Sensors/Write");
    device::CommandEncoder encoder = queue.CreateEncoder();
    {
        const device::ZoneScope scope(encoder, zone);
        for (const auto& [entry, rows] : {std::pair{"WriteCameras", params.cameraRows},
                                          std::pair{"WriteDistortions", params.distortionRows},
                                          std::pair{"WriteDetectors", params.detectorRows},
                                          std::pair{"WriteLidars", params.lidarRows},
                                          std::pair{"WriteRadars", params.radarRows}}) {
            if (rows > 0) {
                encoder.Dispatch(Kernels().Entry(entry), params,
                                 {.x = device::GroupCount(rows, kSensorGroup)});
            }
        }
    }
    queue.Submit(encoder.Finish());
}

void SensorTable::Prepare() {
    Kernels();
}

device::Program& SensorTable::Kernels() {
    if (!m_program) {
        m_program = m_structure.GetColumnStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.tables.kernels.sensors"},
             .library = &kSceneKernels,
             .entries = {"WriteCameras", "WriteDistortions", "WriteDetectors", "WriteLidars",
                         "WriteRadars"},
             .label = "sensors"});
    }
    return *m_program;
}

}

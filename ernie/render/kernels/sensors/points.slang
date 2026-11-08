// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
__exported import ernie.render.kernels.sensors.isensor;

using namespace ernie;

void WritePoint(CapturePixel at, bool returned, SensorFrame frame, float3 position,
                float distance, float intensity, float radialVelocity, uint prim,
                int64_t instance) {
    for (uint64_t points = at.capture->vars[uint(RenderVarSource::PointCloud)]; points != 0;
         points = Ptr<PointCloud>(points)->next) {
        PointCloud* cloud = Ptr<PointCloud>(points);
        Ptr<uint>(cloud->flags)[at.index] = returned ? 1 : 0;
        if (returned) {
            const float3 point =
                cloud->world != 0 ? position + frame.origin : frame.ToLocal(position);
            Ptr<PointRecord>(cloud->records)[at.index] = {
                point.x, point.y, point.z, distance, intensity, radialVelocity, prim, 0, instance
            };
        }
    }
}

void ScatterPoints(ScatterParams p) {
    const uint slot = LinearThreadIndex();
    if (slot >= p.slotCount) {
        return;
    }
    PointCloud* cloud = p.cloud;
    const uint perSensor = cloud->slotsPerSensor;
    const uint layer = slot / perSensor;
    const uint local = slot - layer * perSensor;
    const uint first = layer * perSensor;
    uint* offsets = Ptr<uint>(cloud->offsets);
    uint* flags = Ptr<uint>(cloud->flags);
    const uint base = offsets[first];
    if (local == perSensor - 1) {
        Ptr<uint>(cloud->counts)[layer] = offsets[slot] + flags[slot] - base;
    }
    if (flags[slot] == 0) {
        return;
    }
    const uint at = first + offsets[slot] - base;
    const PointRecord record = Ptr<PointRecord>(cloud->records)[slot];
    if (cloud->coordinates != 0) {
        float* coordinates = Ptr<float>(cloud->coordinates);
        coordinates[3 * at] = record.x;
        coordinates[3 * at + 1] = record.y;
        coordinates[3 * at + 2] = record.z;
    }
    if (cloud->distance != 0) {
        Ptr<float>(cloud->distance)[at] = record.distance;
    }
    if (cloud->intensity != 0) {
        Ptr<float>(cloud->intensity)[at] = record.intensity;
    }
    if (cloud->radialVelocity != 0) {
        Ptr<float>(cloud->radialVelocity)[at] = record.radialVelocity;
    }
    if (cloud->instanceId != 0) {
        Ptr<int64_t>(cloud->instanceId)[at] = record.instanceId;
    }
    if (cloud->stableId != 0) {
        const StableId id = p.tables->semantics.stableIds[record.prim];
        Ptr<uint4>(cloud->stableId)[at] = uint4(id.words[0], id.words[1], id.words[2], id.words[3]);
    }
    const uint echo = local % cloud->returns;
    const uint beam = local / cloud->returns;
    if (cloud->emitter != 0) {
        Ptr<uint>(cloud->emitter)[at] = beam / cloud->width;
    }
    if (cloud->firing != 0) {
        Ptr<uint>(cloud->firing)[at] = beam % cloud->width;
    }
    if (cloud->echo != 0) {
        Ptr<uint>(cloud->echo)[at] = echo;
    }
}

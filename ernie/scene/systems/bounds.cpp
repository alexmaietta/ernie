// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/systems/bounds.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/kernels/bounds.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/kernels/tables/render_tables.h"
#include "ernie/scene/timeline/changes.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

constexpr ColumnRef kInputs[] = {kPointBasedPoints,
                                 kSphereRadius,
                                 kCubeSize,
                                 kPlaneWidth,
                                 kPlaneLength,
                                 kPlaneAxis,
                                 kCylinderRadius,
                                 kCylinderHeight,
                                 kCylinderAxis,
                                 kConeRadius,
                                 kConeHeight,
                                 kConeAxis,
                                 kCapsuleRadius,
                                 kCapsuleHeight,
                                 kCapsuleAxis,
                                 kPointInstancerPrototypes,
                                 kPointInstancerProtoIndices,
                                 kPointInstancerPositions,
                                 kPointInstancerOrientations,
                                 kPointInstancerScales,
                                 kPointInstancerIds,
                                 kPointInstancerInvisibleIds,
                                 kPrototypeAPIPrototype};

uint32_t Ordered(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

}

std::span<const ColumnRef> Bounds::Inputs() const {
    return kInputs;
}

void Bounds::Relayout() {
    const Prims& prims = m_catalog.GetPrims();
    device::Device& device = m_catalog.GetStore().GetDevice();
    const uint32_t worlds = prims.WorldCount();
    std::vector<uint32_t> origins(std::max(worlds, 1u), kNoRow);
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (prims.Alive(prim) && IsA(prims.At(prim).schema, SchemaId::World)) {
            origins[prims.At(prim).world] = prims.Row(prim, SchemaId::Xformable);
        }
    }
    m_origins =
        device.CreateBuffer({.label = "bounds world origins"}, std::span<const uint32_t>(origins));
    if (worlds + 1 != m_slots) {
        m_slots = worlds + 1;
        m_extents = device.CreateBuffer(
            {.bytes = uint64_t(m_slots) * 6 * sizeof(uint32_t), .label = "world extents"});
        m_boxes = device.CreateBuffer(
            {.bytes = uint64_t(m_slots) * sizeof(BvhBox), .label = "world bounds"});
    }
    const uint32_t boundables = std::max(m_catalog.GetPrims().Size(), 1u);
    if (!m_boundables || boundables > m_boundableCapacity) {
        m_boundableCapacity = boundables;
        m_boundables = device.CreateBuffer(
            {.bytes = uint64_t(boundables) * sizeof(BoundableBox), .label = "prim bounds"});
    }
    m_epoch = m_catalog.Epoch();
}

void Bounds::Run(const ChangeSet& changes, TableLog&) {
    ERNIE_ZONE("Bounds/Run");
    Store& store = m_catalog.GetStore();
    device::Device& device = store.GetDevice();
    if (changes.resync || changes.structure || m_epoch != m_catalog.Epoch()) {
        Relayout();
    }
    store.Flush();
    BoundsParams params{};
    params.scene.address = store.DeviceData();
    params.origins.address = m_origins->Address();
    params.extents.address = m_extents->Address();
    params.boxes.address = m_boxes->Address();
    params.transforms = changes.tables->transforms;
    params.imageables = changes.tables->imageables;
    params.meshes = changes.tables->meshes;
    params.boundables.address = m_boundables->Address();
    params.primCount = m_catalog.GetPrims().Size();
    params.slotCount = m_slots;
    const uint64_t instancerBytes = std::max<uint64_t>(params.primCount, 1) * 6 * sizeof(uint32_t);
    if (!m_instancerExtents || m_instancerExtents->Bytes() < instancerBytes) {
        m_instancerExtents = device.CreateBuffer(
            {.bytes = std::bit_ceil(instancerBytes), .label = "instancer extents"});
    }
    params.instancerExtents.address = m_instancerExtents->Address();
    const bool slots = m_instances.Layout() != nullptr;
    if (slots) {
        params.instances.address = m_instances.Layout()->Address();
        params.instanceSlots = uint32_t(m_instances.InstancePrims().size());
        params.tlasCount = uint32_t(m_instances.InstanceCounts().size());
    }

    const uint64_t half = uint64_t(m_slots) * 3 * sizeof(uint32_t);
    device::Queue& queue = device.GetQueue();
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Bounds/Compute");
    device::CommandEncoder encoder = queue.CreateEncoder();
    {
        const device::ZoneScope scope(encoder, zone);
        encoder.Fill(m_extents->Slice(0, half), Ordered(std::numeric_limits<float>::infinity()));
        encoder.Fill(m_extents->Slice(half), Ordered(-std::numeric_limits<float>::infinity()));
        encoder.Dispatch(Kernels().Entry("PrimBounds"), params,
                         {.x = device::GroupCount(params.primCount, kBoundsGroup)});
        if (slots && params.instanceSlots > 0) {
            std::vector<uint32_t> empty(size_t(params.primCount) * 6);
            for (size_t at = 0; at < empty.size(); ++at) {
                empty[at] = at % 6 < 3 ? Ordered(std::numeric_limits<float>::infinity())
                                       : Ordered(-std::numeric_limits<float>::infinity());
            }
            queue.Write(m_instancerExtents->Slice(0, empty.size() * sizeof(uint32_t)),
                        empty.data());
            encoder.Barrier();
            encoder.Dispatch(Kernels().Entry("SlotBounds"), params,
                             {.x = device::GroupCount(params.instanceSlots, kBoundsGroup)});
            encoder.Barrier();
            encoder.Dispatch(Kernels().Entry("InstancerBounds"), params,
                             {.x = device::GroupCount(params.primCount, kBoundsGroup)});
        }
        encoder.Barrier();
        encoder.Dispatch(Kernels().Entry("FinishBoxes"), params,
                         {.x = device::GroupCount(params.slotCount, kBoundsGroup)});
    }
    queue.Submit(encoder.Finish());
}

device::Download Bounds::ReadWorldsAsync() const {
    if (!m_boxes) {
        return {};
    }
    return m_catalog.GetStore().GetDevice().GetQueue().ReadAsync(
        m_boxes->Slice(0, uint64_t(m_slots) * sizeof(BvhBox)));
}

device::Download Bounds::ReadWorldAsync(uint32_t world) const {
    if (!m_boxes) {
        return {};
    }
    const uint32_t slot = WorldSlot(world);
    if (slot >= m_slots) {
        throw Error("The scene has no world {}", world);
    }
    return m_catalog.GetStore().GetDevice().GetQueue().ReadAsync(
        m_boxes->Slice(uint64_t(slot) * sizeof(BvhBox), sizeof(BvhBox)));
}

std::optional<BvhBox> Bounds::World(uint32_t world) const {
    if (!m_boxes) {
        return std::nullopt;
    }
    const uint32_t slot = WorldSlot(world);
    if (slot >= m_slots) {
        throw Error("The scene has no world {}", world);
    }
    BvhBox box{};
    m_catalog.GetStore().GetDevice().GetQueue().Read(
        m_boxes->Slice(uint64_t(slot) * sizeof(BvhBox), sizeof(BvhBox)), &box);
    if (box.lower[0] > box.upper[0]) {
        return std::nullopt;
    }
    return box;
}

void Bounds::Describe(RenderTables& tables) const {
    tables.bounds = m_boxes ? m_boxes->Address() : 0;
    tables.boundables.address = m_boundables ? m_boundables->Address() : 0;
}

void Bounds::Prepare() {
    Kernels();
}

device::Program& Bounds::Kernels() {
    if (!m_program) {
        m_program = m_catalog.GetStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.kernels.bounds"},
             .library = &kSceneKernels,
             .entries = {"PrimBounds", "SlotBounds", "InstancerBounds", "FinishBoxes"},
             .label = "bounds"});
    }
    return *m_program;
}

}

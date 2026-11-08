// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/bounds.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/kernels/bounds.h"
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

std::span<const ColumnRef> BoundsTable::Inputs() const {
    return kInputs;
}

void BoundsTable::Relayout() {
    const Prims& prims = m_structure.GetPrims();
    device::Device& device = m_structure.GetColumnStore().GetDevice();
    const uint32_t worlds = prims.WorldCount();
    std::vector<uint32_t> origins(std::max(worlds, 1u), kNoRow);
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (prims.Alive(prim) && IsA(prims.Record(prim).schema, SchemaId::World)) {
            origins[prims.Record(prim).world] = prims.Row(prim, SchemaId::Xformable);
        }
    }
    m_origins =
        device.CreateBuffer({.label = "bounds world origins"}, std::span<const uint32_t>(origins));
    if (worlds + 1 != m_boxes.Rows()) {
        m_boxes.Allocate(device, worlds + 1);
        m_extents = device.CreateBuffer(
            {.bytes = uint64_t(m_boxes.Rows()) * 6 * sizeof(uint32_t), .label = "world extents"});
    }
    m_primBounds.Reserve(device, prims.Size());
    m_epoch = m_structure.Epoch();
}

void BoundsTable::Run(const ChangeSet& changes, TableVersions&) {
    ERNIE_ZONE("Bounds/Run");
    ColumnStore& store = m_structure.GetColumnStore();
    device::Device& device = store.GetDevice();
    if (changes.resync || changes.structure || m_epoch != m_structure.Epoch()) {
        Relayout();
    }
    BoundsParams params{};
    params.scene.address = store.DeviceData();
    params.origins.address = m_origins->Address();
    params.extents.address = m_extents->Address();
    params.boxes.address = m_boxes.Address();
    params.transforms = changes.tables->transforms;
    params.imageables = changes.tables->imageables;
    params.meshes = changes.tables->meshes;
    params.primBounds.address = m_primBounds.Address();
    params.primCount = m_structure.GetPrims().Size();
    params.slotCount = m_boxes.Rows();
    const uint64_t instancerBytes = std::max<uint64_t>(params.primCount, 1) * 6 * sizeof(uint32_t);
    if (!m_instancerExtents || m_instancerExtents->Bytes() < instancerBytes) {
        m_instancerExtents = device.CreateBuffer(
            {.bytes = std::bit_ceil(instancerBytes), .label = "instancer extents"});
    }
    params.instancerExtents.address = m_instancerExtents->Address();
    const bool slots = changes.tables->instances.address != 0;
    if (slots) {
        params.instances = changes.tables->instances;
        params.instanceSlots = uint32_t(m_instances.InstancePrims().size());
        params.tlasCount = uint32_t(m_instances.InstanceCounts().size());
    }

    const uint64_t half = uint64_t(m_boxes.Rows()) * 3 * sizeof(uint32_t);
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
            const uint64_t lowers = uint64_t(params.primCount) * 3 * sizeof(uint32_t);
            encoder.Fill(m_instancerExtents->Slice(0, lowers),
                         Ordered(std::numeric_limits<float>::infinity()));
            encoder.Fill(m_instancerExtents->Slice(lowers, lowers),
                         Ordered(-std::numeric_limits<float>::infinity()));
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

device::Download BoundsTable::ReadWorldsAsync() const {
    if (!m_boxes.Buffer()) {
        return {};
    }
    return m_structure.GetColumnStore().GetDevice().GetQueue().ReadAsync(
        m_boxes.Buffer()->Slice(0, uint64_t(m_boxes.Rows()) * sizeof(BvhBox)));
}

device::Download BoundsTable::ReadWorldAsync(uint32_t world) const {
    if (!m_boxes.Buffer()) {
        return {};
    }
    const uint32_t slot = WorldSlot(world);
    if (slot >= m_boxes.Rows()) {
        throw Error("The scene has no world {}", world);
    }
    return m_structure.GetColumnStore().GetDevice().GetQueue().ReadAsync(
        m_boxes.Buffer()->Slice(uint64_t(slot) * sizeof(BvhBox), sizeof(BvhBox)));
}

std::optional<BvhBox> BoundsTable::World(uint32_t world) const {
    if (!m_boxes.Buffer()) {
        return std::nullopt;
    }
    const uint32_t slot = WorldSlot(world);
    if (slot >= m_boxes.Rows()) {
        throw Error("The scene has no world {}", world);
    }
    BvhBox box{};
    m_structure.GetColumnStore().GetDevice().GetQueue().Read(
        m_boxes.Buffer()->Slice(uint64_t(slot) * sizeof(BvhBox), sizeof(BvhBox)), &box);
    if (box.lower[0] > box.upper[0]) {
        return std::nullopt;
    }
    return box;
}

void BoundsTable::Describe(RenderTables& tables) const {
    tables.primBounds.address = m_primBounds.Address();
}

void BoundsTable::Prepare() {
    Kernels();
}

device::Program& BoundsTable::Kernels() {
    if (!m_program) {
        m_program = m_structure.GetColumnStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.tables.kernels.bounds"},
             .library = &kSceneKernels,
             .entries = {"PrimBounds", "SlotBounds", "InstancerBounds", "FinishBoxes"},
             .label = "bounds"});
    }
    return *m_program;
}

}

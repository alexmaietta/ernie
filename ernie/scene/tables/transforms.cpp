// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/transforms.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/kernels/transforms.h"
#include "ernie/stage/publish/change_set.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

constexpr ColumnRef kInputs[] = {kXformableTranslate,
                                 kXformableOrient,
                                 kXformableScale,
                                 kXformableResetXformStack,
                                 kImageableVisibility,
                                 kImageablePurpose,
                                 kMaterialBindingAPIMaterialBinding,
                                 kMotionAPIMotionVelocity,
                                 kMotionAPIMotionAngularVelocity};

constexpr TableId kTableOutputs[] = {TableId::Transforms, TableId::Imageables};

}

std::span<const ColumnRef> TransformTable::Inputs() const {
    return kInputs;
}

std::span<const TableId> TransformTable::TableOutputs() const {
    return kTableOutputs;
}

void TransformTable::Describe(RenderTables& tables) const {
    tables.transforms.address = m_transforms.Address();
    tables.imageables.address = m_imageables.Address();
    tables.motions.address = m_motions.Address();
}

void TransformTable::Hold(std::vector<Ref<const RefCounted>>& held) const {
    HoldBuffers(held, {m_transforms.Buffer(), m_imageables.Buffer(), m_motions.Buffer()});
}

uint32_t TransformTable::ParentRow(uint32_t prim) const {
    const Prims& prims = m_structure.GetPrims();
    uint32_t up = prims.Record(prim).parent;
    while (up != kNoPrim && !IsA(prims.Record(up).schema, SchemaId::Imageable)) {
        up = prims.Record(up).parent;
    }
    return up == kNoPrim ? kNoRow : prims.Row(up, SchemaId::Imageable);
}

void TransformTable::Rebuild() {
    ERNIE_ZONE("Hierarchy/Rebuild");
    const Prims& prims = m_structure.GetPrims();
    ColumnStore& store = m_structure.GetColumnStore();
    device::Device& device = store.GetDevice();
    const uint32_t capacity = std::max(store.Schema(SchemaId::Imageable).Capacity(), 1u);
    m_transforms.Allocate(device, store.Schema(SchemaId::Xformable).Capacity());
    m_imageables.Allocate(device, capacity);
    m_parents.Assign(capacity, kNoRow);
    m_xforms.Assign(capacity, kNoRow);
    m_rows.clear();
    std::vector<std::vector<uint32_t>> segments(prims.WorldCount() + 1);
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim) || !IsA(prims.Record(prim).schema, SchemaId::Imageable)) {
            continue;
        }
        const PrimRecord& record = prims.Record(prim);
        const uint32_t row = prims.Row(prim, SchemaId::Imageable);
        m_parents.Set(row, ParentRow(prim));
        if (IsA(record.schema, SchemaId::Xformable)) {
            m_xforms.Set(row, prims.Row(prim, SchemaId::Xformable));
        }
        segments[WorldSlot(record.world)].push_back(row);
        m_rows[prim] = row;
    }
    m_segments.Assign(std::move(segments));

    m_parents.Upload(device);
    m_xforms.Upload(device);
    m_segmentRows = device.CreateBuffer(
        {.label = "hierarchy segment rows"},
        std::vector<uint32_t>(m_segments.Items().begin(), m_segments.Items().end()));
    m_segmentStart = device.CreateBuffer(
        {.label = "hierarchy segment starts"},
        std::vector<uint32_t>(m_segments.Starts().begin(), m_segments.Starts().end()));
    m_segmentCount = device.CreateBuffer(
        {.label = "hierarchy segment counts"},
        std::vector<uint32_t>(m_segments.Counts().begin(), m_segments.Counts().end()));
    m_built = true;
    ++m_rebuilds;
    Log(LogLevel::Debug, "rebuilt the hierarchy: {} rows in {} segments", m_rows.size(),
        m_segments.Size());
}

bool TransformTable::Apply(const ChangeSet& changes) {
    ERNIE_ZONE("Hierarchy/Apply");
    const Prims& prims = m_structure.GetPrims();
    if (m_segments.Size() != prims.WorldCount() + 1) {
        return false;
    }
    std::map<uint32_t, std::vector<uint32_t>> touched;
    const auto segment = [&](uint32_t index) -> std::vector<uint32_t>& {
        auto [entry, fresh] = touched.try_emplace(index);
        if (fresh) {
            const std::span<const uint32_t> items = m_segments.Items(index);
            entry->second.assign(items.begin(), items.end());
        }
        return entry->second;
    };
    for (const StructureEntry& entry : changes.removed) {
        const auto found = m_rows.find(entry.prim);
        if (found == m_rows.end()) {
            continue;
        }
        const uint32_t row = found->second;
        std::erase(segment(WorldSlot(entry.world)), row);
        m_parents.Set(row, kNoRow);
        m_xforms.Set(row, kNoRow);
        m_rows.erase(found);
    }
    for (const StructureEntry& entry : changes.added) {
        if (!prims.Alive(entry.prim) || prims.Generation(entry.prim) != entry.generation ||
            !IsA(prims.Record(entry.prim).schema, SchemaId::Imageable)) {
            continue;
        }
        const uint32_t row = prims.Row(entry.prim, SchemaId::Imageable);
        if (row >= m_parents.Rows()) {
            return false;
        }
        if (IsA(prims.Record(entry.prim).schema, SchemaId::Xformable) &&
            prims.Row(entry.prim, SchemaId::Xformable) >= m_transforms.Rows()) {
            return false;
        }
        m_parents.Set(row, ParentRow(entry.prim));
        m_xforms.Set(row, IsA(prims.Record(entry.prim).schema, SchemaId::Xformable)
                              ? prims.Row(entry.prim, SchemaId::Xformable)
                              : kNoRow);
        segment(WorldSlot(entry.world)).push_back(row);
        m_rows[entry.prim] = row;
    }
    for (const auto& [index, items] : touched) {
        if (!m_segments.Replace(index, items)) {
            return false;
        }
    }

    device::Device& device = m_structure.GetColumnStore().GetDevice();
    m_parents.Upload(device);
    m_xforms.Upload(device);
    device::Queue& queue = device.GetQueue();
    for (const auto& [index, items] : touched) {
        const uint32_t start = m_segments.Starts()[index];
        queue.Write(m_segmentRows->Slice(start * sizeof(uint32_t),
                                         m_segments.Capacity(index) * sizeof(uint32_t)),
                    m_segments.Items().data() + start);
    }
    if (!touched.empty()) {
        queue.Write(*m_segmentCount, m_segments.Counts().data());
    }
    return true;
}

void TransformTable::Run(const ChangeSet& changes, TableVersions& versions) {
    ERNIE_ZONE("Hierarchy/Run");
    ColumnStore& store = m_structure.GetColumnStore();
    const bool rebuild = changes.resync || !m_built || (changes.structure && !Apply(changes));
    if (rebuild) {
        Rebuild();
    }
    WriteMotions(rebuild || !changes.apis.empty());

    std::vector<uint32_t> segments;
    const bool global = rebuild || changes.Global();
    if (global) {
        segments.resize(m_segments.Size());
        std::iota(segments.begin(), segments.end(), 0u);
    } else {
        for (const uint32_t world : changes.worlds) {
            segments.push_back(WorldSlot(world));
        }
    }
    std::erase_if(segments, [&](uint32_t segment) {
        return segment >= m_segments.Size() || m_segments.Counts()[segment] == 0;
    });
    m_dispatched = 0;
    if (segments.empty()) {
        return;
    }
    uint32_t widest = 0;
    for (const uint32_t segment : segments) {
        widest = std::max(widest, m_segments.Counts()[segment]);
        m_dispatched += m_segments.Counts()[segment];
    }

    const Ref<device::Buffer> dirty =
        store.GetDevice().CreateBuffer({.label = "hierarchy dirty segments"}, segments);
    ResolveParams params{};
    params.tree.parent.address = m_parents.Buffer()->Address();
    params.tree.xform.address = m_xforms.Buffer()->Address();
    params.tree.segmentRows.address = m_segmentRows->Address();
    params.tree.segmentStart.address = m_segmentStart->Address();
    params.tree.segmentCount.address = m_segmentCount->Address();
    params.scene.address = store.DeviceData();
    params.segments.address = dirty->Address();
    params.transforms.address = m_transforms.Address();
    params.imageables.address = m_imageables.Address();
    params.since = changes.sinceStamp;
    params.until = changes.untilStamp;
    params.all = rebuild ? 1 : 0;
    params.inherit = changes.removed.empty() && changes.apis.empty() ? 0 : 1;

    const auto written = [&](std::span<const ColumnRef> inputs) {
        return rebuild || std::ranges::any_of(inputs, [&](ColumnRef input) {
                   return store.LastWrite(input) > m_seen;
               });
    };
    constexpr ColumnRef kTransforms[] = {kXformableTranslate, kXformableOrient, kXformableScale,
                                         kXformableResetXformStack};
    constexpr ColumnRef kInherited[] = {kImageableVisibility, kImageablePurpose,
                                        kMaterialBindingAPIMaterialBinding};
    const bool moved = written(kTransforms);
    const bool inherited = written(kInherited) || !changes.removed.empty() || !changes.apis.empty();
    m_seen = store.WriteSequence();
    std::vector<uint32_t> worlds;
    if (global) {
        worlds.push_back(kGlobalWorld);
    } else {
        for (const uint32_t segment : segments) {
            worlds.push_back(SlotWorld(segment));
        }
    }
    if (moved) {
        versions.Record(TableId::Transforms, worlds);
    }
    if (inherited) {
        versions.Record(TableId::Imageables, worlds);
    }
    device::Queue& queue = store.GetDevice().GetQueue();
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Hierarchy/Resolve");
    device::CommandEncoder encoder = queue.CreateEncoder();
    {
        const device::ZoneScope scope(encoder, zone);
        encoder.Dispatch(
            Kernels().Entry("ResolveRows"), params,
            {.x = device::GroupCount(widest, kHierarchyGroup), .y = uint32_t(segments.size())});
    }
    queue.Submit(encoder.Finish());
}

void TransformTable::WriteMotions(bool restructured) {
    ColumnStore& store = m_structure.GetColumnStore();
    const bool written = store.LastWrite(kMotionAPIMotionVelocity) > m_motionSeen ||
                         store.LastWrite(kMotionAPIMotionAngularVelocity) > m_motionSeen;
    m_motionSeen = store.WriteSequence();
    device::Device& device = store.GetDevice();
    const bool grown = m_motions.Reserve(device, store.Schema(SchemaId::MotionAPI).Capacity());
    const uint32_t rows = store.Schema(SchemaId::MotionAPI).Rows();
    if (rows == 0 || !(written || restructured || grown)) {
        return;
    }
    MotionParams params{};
    params.scene.address = store.DeviceData();
    params.motions.address = m_motions.Address();
    params.rows = rows;
    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.Dispatch(Kernels().Entry("WriteMotions"), params,
                     {.x = device::GroupCount(rows, kHierarchyGroup)});
    queue.Submit(encoder.Finish());
}

void TransformTable::Prepare() {
    Kernels();
}

device::Program& TransformTable::Kernels() {
    if (!m_program) {
        m_program = m_structure.GetColumnStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.tables.kernels.transforms"},
             .library = &kSceneKernels,
             .entries = {"ResolveRows", "WriteMotions"},
             .label = "hierarchy"});
    }
    return *m_program;
}

}

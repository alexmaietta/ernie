// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/points.h"

#include <algorithm>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/kernels/points.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

constexpr ColumnRef kInputs[] = {kPointBasedPoints, kPointBasedNormals, kPointsWidths, kPointsIds};

constexpr TableId kTableOutputs[] = {TableId::Points, TableId::PointsMoved};

}

std::span<const ColumnRef> PointsTable::Inputs() const {
    return kInputs;
}

std::span<const TableId> PointsTable::TableOutputs() const {
    return kTableOutputs;
}

void PointsTable::Describe(RenderTables& tables) const {
    tables.points.address = m_records.Buffer() ? m_records.Buffer()->Address() : 0;
}

void PointsTable::Place(uint32_t prim) {
    const Prims& prims = m_structure.GetPrims();
    Cloud cloud;
    cloud.prim = prim;
    cloud.generation = prims.Generation(prim);
    cloud.pointBasedRow = prims.Row(prim, SchemaId::PointBased);
    cloud.row = prims.Row(prim, SchemaId::Points);
    m_byPointBasedRow[cloud.pointBasedRow] = prim;
    m_byRow[cloud.row] = prim;
    Link(m_clouds[prim] = std::move(cloud));
}

void PointsTable::Forget(uint32_t prim) {
    const Cloud& cloud = m_clouds.at(prim);
    if (cloud.bounds.count > 0) {
        m_structure.GetColumnStore().Arrays(ColumnType::Float3Array).Free(cloud.bounds);
    }
    m_records.Set(cloud.row, PointsRecord{});
    m_byPointBasedRow.erase(cloud.pointBasedRow);
    m_byRow.erase(cloud.row);
    m_clouds.erase(prim);
}

void PointsTable::Link(Cloud& cloud) {
    ColumnStore& store = m_structure.GetColumnStore();
    const ArrayRef points = store.Array(kPointBasedPoints, cloud.pointBasedRow);
    const ArrayRef widths = store.Array(kPointsWidths, cloud.row);
    const ArrayRef normals = store.Array(kPointBasedNormals, cloud.pointBasedRow);
    const ArrayRef ids = store.Array(kPointsIds, cloud.row);
    if (points.count > cloud.capacity || points.count < cloud.capacity / 4) {
        cloud.capacity = points.count > cloud.capacity
                             ? std::max(points.count, cloud.capacity + cloud.capacity / 2)
                             : points.count;
        cloud.blas = nullptr;
        cloud.boxes = nullptr;
        if (cloud.capacity > 0) {
            device::Device& device = store.GetDevice();
            cloud.boxes = device.CreateBuffer({.bytes = uint64_t(cloud.capacity) * sizeof(BvhBox),
                                               .usage = device::BufferUsage::AccelBuildInput,
                                               .label = "point boxes"});
            cloud.blas = m_builder.CreateBlas(device::AccelBoxes{
                .boxes = cloud.boxes->Slice(0), .count = cloud.capacity, .refittable = true});
        }
        cloud.fresh = true;
    }
    if (points.count != cloud.count) {
        cloud.count = points.count;
        cloud.fresh = true;
    }
    if (cloud.bounds.count == 0) {
        cloud.bounds = store.Arrays(ColumnType::Float3Array).Allocate(2);
    }
    cloud.update = points.count > 0;
    PointsRecord record{};
    if (cloud.blas && points.count > 0) {
        record.blas = cloud.blas->Handle();
        record.points = points.address;
        record.widths = widths.address;
        record.pointCount = points.count;
        record.widthCount = widths.count;
        record.normals = normals.address;
        record.normalCount = normals.count;
        record.ids = ids.address;
        record.idCount = ids.count;
        record.bounds = cloud.bounds;
    }
    const Prims& prims = m_structure.GetPrims();
    if (m_records.Set(cloud.row, record) && prims.Alive(cloud.prim)) {
        m_relinked.push_back(prims.Record(cloud.prim).world);
    }
}

void PointsTable::Run(const ChangeSet& changes, TableVersions& versions) {
    ERNIE_ZONE("Points/Run");
    const Prims& prims = m_structure.GetPrims();
    const auto isPoints = [&](uint32_t prim) {
        return prims.Alive(prim) && IsA(prims.Record(prim).schema, SchemaId::Points);
    };
    if (changes.resync) {
        while (!m_clouds.empty()) {
            Forget(m_clouds.begin()->first);
        }
        for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
            if (isPoints(prim)) {
                Place(prim);
            }
        }
    } else {
        for (const StructureEntry& entry : changes.removed) {
            const auto found = m_clouds.find(entry.prim);
            if (found != m_clouds.end() && found->second.generation == entry.generation) {
                Forget(entry.prim);
            }
        }
        for (const StructureEntry& entry : changes.added) {
            if (isPoints(entry.prim) && prims.Generation(entry.prim) == entry.generation) {
                if (m_clouds.contains(entry.prim)) {
                    Forget(entry.prim);
                }
                Place(entry.prim);
            }
        }
        const auto relink = [&](ColumnRef column, const auto& owners) {
            for (const uint32_t row : changes.ArrayRows(column)) {
                const auto owner = owners.find(row);
                if (owner != owners.end()) {
                    Cloud& cloud = m_clouds.at(owner->second);
                    if (!cloud.update) {
                        Link(cloud);
                    }
                }
            }
        };
        relink(kPointBasedPoints, m_byPointBasedRow);
        relink(kPointBasedNormals, m_byPointBasedRow);
        relink(kPointsWidths, m_byRow);
        relink(kPointsIds, m_byRow);
    }
    std::vector<uint32_t> updated;
    Build(updated);
    m_records.Reserve(m_structure.GetColumnStore().Schema(SchemaId::Points).Capacity());
    if (m_records.Upload(m_structure.GetColumnStore().GetDevice())) {
        m_relinked = {kGlobalWorld};
    }
    if (!m_relinked.empty()) {
        versions.Record(TableId::Points, m_relinked);
    }
    if (!updated.empty()) {
        versions.Record(TableId::PointsMoved, updated);
    }
    m_relinked.clear();
}

void PointsTable::Build(std::vector<uint32_t>& updated) {
    std::vector<PointsJob> jobs;
    std::vector<Ref<bvh::Blas>> built;
    std::vector<Ref<bvh::Blas>> refit;
    uint32_t total = 0;
    uint32_t active = 0;
    const Prims& prims = m_structure.GetPrims();
    for (auto& [prim, cloud] : m_clouds) {
        if (!cloud.update) {
            continue;
        }
        cloud.update = false;
        if (cloud.fresh || cloud.refits >= m_maxRefits) {
            cloud.fresh = false;
            cloud.refits = 0;
            built.push_back(cloud.blas);
        } else {
            ++cloud.refits;
            refit.push_back(cloud.blas);
        }
        PointsJob job{};
        job.record = m_records.Row(cloud.row);
        job.boxes.address = cloud.boxes->Address();
        job.bounds.address = cloud.bounds.address;
        job.first = total;
        job.capacity = cloud.capacity;
        jobs.push_back(job);
        total += cloud.capacity;
        active += cloud.count;
        if (prims.Alive(prim)) {
            updated.push_back(prims.Record(prim).world);
        }
    }
    m_stats.clouds = uint32_t(m_clouds.size());
    m_stats.built = uint32_t(built.size());
    m_stats.refit = uint32_t(refit.size());
    m_stats.capacity = total;
    m_stats.points = active;
    if (jobs.empty()) {
        return;
    }
    ColumnStore& store = m_structure.GetColumnStore();
    device::Device& device = store.GetDevice();
    store.Flush();
    const Ref<device::Buffer> table =
        device.CreateBuffer({.label = "points jobs"}, std::span<const PointsJob>(jobs));
    const uint64_t extentWords = uint64_t(jobs.size()) * 3;
    const Ref<device::Buffer> extents = device.CreateBuffer(
        {.bytes = extentWords * 2 * sizeof(uint32_t), .label = "points extents"});
    PointsParams params{};
    params.jobs.address = table->Address();
    params.extents.address = extents->Address();
    params.jobCount = uint32_t(jobs.size());
    params.total = total;
    static const profiler::ZoneId boxesZone = profiler::ZoneId::Named("Points/Boxes");
    static const profiler::ZoneId blasZone = profiler::ZoneId::Named("Points/Blas");
    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.BeginZone(boxesZone);
    encoder.Fill(extents->Slice(0, extentWords * sizeof(uint32_t)), ~0u);
    encoder.Fill(extents->Slice(extentWords * sizeof(uint32_t)), 0u);
    encoder.Dispatch(Kernels().Entry("WritePointBoxes"), params,
                     {.x = device::GroupCount(total, kPointsGroup)});
    encoder.Dispatch(Kernels().Entry("FinishPointBounds"), params,
                     {.x = device::GroupCount(params.jobCount, kPointsGroup)});
    encoder.EndZone();
    encoder.BeginZone(blasZone);
    m_builder.Build(encoder, built);
    m_builder.Refit(encoder, refit);
    encoder.EndZone();
    queue.Submit(encoder.Finish());
}

void PointsTable::Hold(std::vector<Ref<const RefCounted>>& held) const {
    HoldBuffers(held, {m_records.Buffer()});
    for (const auto& [prim, cloud] : m_clouds) {
        if (cloud.blas) {
            held.emplace_back(cloud.blas);
            held.emplace_back(cloud.boxes);
        }
    }
}

void PointsTable::Prepare() {
    Kernels();
}

device::Program& PointsTable::Kernels() {
    if (!m_program) {
        m_program = m_structure.GetColumnStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.tables.kernels.points"},
             .library = &kSceneKernels,
             .entries = {"WritePointBoxes", "FinishPointBounds"},
             .label = "points"});
    }
    return *m_program;
}

}

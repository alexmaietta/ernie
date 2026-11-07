// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/systems/geometry.h"

#include <algorithm>
#include <bit>
#include <span>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/kernels/geometry.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/kernels/tables/render_tables.h"
#include "ernie/scene/timeline/changes.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

constexpr uint32_t kPreparedScan = 1024;

constexpr ColumnRef kInputs[] = {kPointBasedPoints, kMeshFaceVertexIndices, kMeshFaceVertexCounts};

constexpr TableId kTableOutputs[] = {TableId::Meshes, TableId::MeshData};

void AddWorld(std::vector<uint32_t>& worlds, uint32_t world) {
    if (std::ranges::find(worlds, world) == worlds.end()) {
        worlds.push_back(world);
    }
}

}

std::span<const ColumnRef> Geometry::Inputs() const {
    return kInputs;
}

std::span<const TableId> Geometry::TableOutputs() const {
    return kTableOutputs;
}

void Geometry::Describe(RenderTables& tables) const {
    tables.meshes.address = m_records.Buffer() ? m_records.Buffer()->Address() : 0;
}

std::optional<MeshShape> Geometry::Find(uint32_t prim) const {
    const auto found = m_meshes.find(prim);
    if (found == m_meshes.end() || !found->second.key) {
        return std::nullopt;
    }
    const Shape& shape = m_shapes.at(*found->second.key);
    const Topology& topology = m_topologies.at(found->second.key->topology);
    return MeshShape{shape.blas, shape.pointCount, topology.triangleCount, shape.users};
}

void Geometry::Triangulate() {
    std::vector<Topology*> pending;
    std::vector<TopologyJob> jobs;
    uint32_t faces = 0;
    for (auto& [key, topology] : m_topologies) {
        if (topology.ready) {
            continue;
        }
        TopologyJob job{};
        job.indices.address = key.indices;
        job.counts.address = key.counts;
        job.indexCount = key.indexCount;
        job.faceCount = key.faceCount;
        job.firstFace = faces;
        faces += key.faceCount;
        jobs.push_back(job);
        pending.push_back(&topology);
    }
    if (jobs.empty()) {
        return;
    }
    ERNIE_ZONE("Geometry/Triangulate");
    m_stats.triangulated += uint32_t(jobs.size());
    Store& store = m_catalog.GetStore();
    device::Device& device = store.GetDevice();
    device::Queue& queue = device.GetQueue();
    const auto scratch = [&](uint64_t words, const char* label) {
        return device.CreateBuffer({.bytes = std::max<uint64_t>(words, 1) * 4, .label = label});
    };
    const auto scan = [&](uint32_t count) -> const algorithms::Scan& {
        if (count > m_scanCapacity) {
            m_scanCapacity = std::bit_ceil(count);
            m_scan = std::make_unique<algorithms::Scan>(device, m_scanCapacity);
        }
        return *m_scan;
    };
    const auto dispatch = [&](device::CommandEncoder& encoder, const char* entry,
                              const TriangulateParams& params, uint32_t threads) {
        encoder.Dispatch(Kernels().Entry(entry), params,
                         {.x = device::GroupCount(threads, kTopologyGroup)});
    };

    const Ref<device::Buffer> jobBuffer =
        device.CreateBuffer({.label = "topology jobs"}, std::span<const TopologyJob>(jobs));
    const Ref<device::Buffer> work = scratch(faces, "face work");
    const Ref<device::Buffer> faceStarts = scratch(faces, "face starts");
    const Ref<device::Buffer> triangleStarts = scratch(faces, "face triangle starts");
    const Ref<device::Buffer> sizes = scratch(jobs.size() * 2, "topology sizes");
    TriangulateParams params{};
    params.jobs.address = jobBuffer->Address();
    params.work.address = work->Address();
    params.faceStarts.address = faceStarts->Address();
    params.triangleStarts.address = triangleStarts->Address();
    params.sizes.address = sizes->Address();
    params.jobCount = uint32_t(jobs.size());
    params.faceCount = faces;
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Geometry/Triangulate");
    {
        device::CommandEncoder encoder = queue.CreateEncoder();
        {
            const device::ZoneScope scope(encoder, zone);
            encoder.Fill(*sizes);
            dispatch(encoder, "FaceSizes", params, faces);
            scan(faces).ExclusiveSum(encoder, *work, *faceStarts, faces);
            dispatch(encoder, "FaceTriangles", params, faces);
            scan(faces).ExclusiveSum(encoder, *work, *triangleStarts, faces);
            dispatch(encoder, "TopologySizes", params, params.jobCount);
        }
        queue.Submit(encoder.Finish());
    }
    std::vector<uint32_t> counted(jobs.size() * 2);
    queue.Read(sizes->Slice(0, counted.size() * 4), counted.data());

    ArrayHeap& ints = store.Arrays(ColumnType::IntArray);
    std::vector<TopologyJob> emitted;
    uint32_t corners = 0;
    uint32_t points = 0;
    for (uint32_t i = 0; i < jobs.size(); ++i) {
        Topology& topology = *pending[i];
        topology.ready = true;
        topology.triangleCount = counted[i * 2];
        topology.covered = counted[i * 2 + 1];
        if (topology.triangleCount == 0) {
            continue;
        }
        const uint32_t count = topology.triangleCount;
        topology.triangles = ints.Allocate(count * 3);
        topology.corners = ints.Allocate(count * 3);
        topology.faces = ints.Allocate(count);
        topology.starts = ints.Allocate(topology.covered + 1);
        topology.adjacency = ints.Allocate(count * 3);
        TopologyJob& job = jobs[i];
        job.triangles.address = topology.triangles.address;
        job.corners.address = topology.corners.address;
        job.faces.address = topology.faces.address;
        job.starts.address = topology.starts.address;
        job.adjacency.address = topology.adjacency.address;
        job.triangleCount = count;
        job.covered = topology.covered;
        job.firstCorner = corners;
        job.firstPoint = points;
        corners += count * 3;
        points += topology.covered;
        emitted.push_back(job);
    }
    if (emitted.empty()) {
        return;
    }
    if (corners > m_sortCapacity) {
        m_sortCapacity = std::bit_ceil(corners);
        m_sort = std::make_unique<algorithms::RadixSort>(device, m_sortCapacity, 4);
    }
    const Ref<device::Buffer> updated =
        device.CreateBuffer({.label = "topology jobs"}, std::span<const TopologyJob>(jobs));
    const Ref<device::Buffer> emittedBuffer =
        device.CreateBuffer({.label = "emitted topologies"}, std::span<const TopologyJob>(emitted));
    const Ref<device::Buffer> keys = scratch(corners, "corner points");
    const Ref<device::Buffer> values = scratch(corners, "corner order");
    const Ref<device::Buffer> histogram = scratch(points, "point triangle counts");
    const Ref<device::Buffer> pointStarts = scratch(points, "point starts");
    params.jobs.address = updated->Address();
    params.emitted.address = emittedBuffer->Address();
    params.keys.address = keys->Address();
    params.values.address = values->Address();
    params.histogram.address = histogram->Address();
    params.pointStarts.address = pointStarts->Address();
    params.emittedCount = uint32_t(emitted.size());
    params.cornerCount = corners;
    params.pointCount = points;
    device::CommandEncoder encoder = queue.CreateEncoder();
    {
        const device::ZoneScope scope(encoder, zone);
        encoder.Fill(*histogram);
        dispatch(encoder, "EmitTriangles", params, faces);
        m_sort->Sort(encoder, *keys, *values, corners,
                     std::max<int>(1, std::bit_width(points - 1)));
        scan(points).ExclusiveSum(encoder, *histogram, *pointStarts, points);
        dispatch(encoder, "WriteAdjacency", params, corners);
        dispatch(encoder, "WriteStarts", params, points);
    }
    queue.Submit(encoder.Finish());
}

void Geometry::Prune() {
    if (!m_stale) {
        return;
    }
    m_stale = false;
    Store& store = m_catalog.GetStore();
    ArrayHeap& ints = store.Arrays(ColumnType::IntArray);
    std::erase_if(m_shapes, [&](const auto& entry) {
        if (entry.second.users > 0) {
            return false;
        }
        store.Arrays(ColumnType::Float3Array).Free(entry.second.normals);
        store.Arrays(ColumnType::Float3Array).Free(entry.second.bounds);
        return true;
    });
    std::erase_if(m_topologies, [&](const auto& entry) {
        if (entry.second.users > 0) {
            return false;
        }
        const Topology& topology = entry.second;
        for (const ArrayRef& array : {topology.triangles, topology.corners, topology.faces,
                                      topology.starts, topology.adjacency}) {
            ints.Free(array);
        }
        return true;
    });
}

void Geometry::Settle() {
    m_stale = m_stale || !m_placing.empty();
    for (const auto& placing : m_placing) {
        m_topologies.try_emplace(placing.second.topology);
    }
    Triangulate();
    Store& store = m_catalog.GetStore();
    for (const auto& [prim, key] : m_placing) {
        Mesh& mesh = m_meshes.at(prim);
        Topology& topology = m_topologies.at(key.topology);
        if (topology.triangleCount > 0 && topology.covered <= key.pointCount) {
            const auto [entry, fresh] = m_shapes.try_emplace(key);
            Shape& shape = entry->second;
            if (fresh) {
                const ArrayRef points = store.Array(kPointBasedPoints, mesh.pointsRow);
                ArrayHeap& float3s = store.Arrays(ColumnType::Float3Array);
                shape.pointCount = points.count;
                shape.normals = float3s.Allocate(points.count);
                shape.bounds = float3s.Allocate(2);
                shape.blas = m_builder.CreateBlas(device::AccelTriangles{
                    .vertices =
                        float3s.Range(points).Slice(0, uint64_t(points.count) * sizeof(Float3)),
                    .stride = sizeof(Float3),
                    .vertexCount = points.count,
                    .indices = store.Arrays(ColumnType::IntArray)
                                   .Range(topology.triangles)
                                   .Slice(0, uint64_t(topology.triangleCount) * 12),
                    .count = topology.triangleCount,
                    .refittable = true});
                shape.build = true;
            }
            ++shape.users;
            ++topology.users;
            mesh.key = key;
            mesh.shape = &shape;
        }
        Link(mesh);
    }
    m_placing.clear();
}

void Geometry::Place(uint32_t prim) {
    const Prims& prims = m_catalog.GetPrims();
    Mesh mesh;
    mesh.prim = prim;
    mesh.generation = prims.Generation(prim);
    mesh.pointsRow = prims.Row(prim, SchemaId::PointBased);
    mesh.meshRow = prims.Row(prim, SchemaId::Mesh);
    m_byPointsRow[mesh.pointsRow] = prim;
    m_byMeshRow[mesh.meshRow] = prim;

    m_meshes[prim] = mesh;
    const ShapeKey key = Key(mesh);
    if (key.pointCount == 0 || key.topology.indexCount == 0 || key.topology.faceCount == 0) {
        Link(mesh);
        return;
    }
    m_topologies.try_emplace(key.topology);
    m_placing.emplace_back(prim, key);
}

Geometry::ShapeKey Geometry::Key(const Mesh& mesh) const {
    const Store& store = m_catalog.GetStore();
    const ArrayRef points = store.Array(kPointBasedPoints, mesh.pointsRow);
    const ArrayRef indices = store.Array(kMeshFaceVertexIndices, mesh.meshRow);
    const ArrayRef counts = store.Array(kMeshFaceVertexCounts, mesh.meshRow);
    return {points.address,
            points.count,
            {indices.address, indices.count, counts.address, counts.count}};
}

void Geometry::Release(Mesh& mesh) {
    if (mesh.key) {
        --m_shapes.at(*mesh.key).users;
        --m_topologies.at(mesh.key->topology).users;
        mesh.key.reset();
        mesh.shape = nullptr;
        m_stale = true;
    }
}

void Geometry::Link(const Mesh& mesh) {
    MeshRecord record{};
    if (mesh.key) {
        const Shape& shape = m_shapes.at(*mesh.key);
        const Topology& topology = m_topologies.at(mesh.key->topology);
        record.blas = shape.blas->Handle();
        record.corners = topology.corners;
        record.faces = topology.faces;
        record.starts = topology.starts;
        record.adjacency = topology.adjacency;
        record.normals = shape.normals;
        record.bounds = shape.bounds;
    }
    if (m_records.Set(mesh.meshRow, record) && m_catalog.GetPrims().Alive(mesh.prim)) {
        AddWorld(m_relinked, m_catalog.GetPrims().At(mesh.prim).world);
    }
}

void Geometry::Run(const ChangeSet& changes, TableLog& log) {
    ERNIE_ZONE("Geometry/Run");
    const Prims& prims = m_catalog.GetPrims();
    m_stats.built = 0;
    m_stats.refit = 0;
    m_stats.triangulated = 0;
    const auto isMesh = [&](uint32_t prim) {
        return prims.Alive(prim) && IsA(prims.At(prim).schema, SchemaId::Mesh);
    };
    const auto forget = [&](uint32_t prim) {
        Mesh& mesh = m_meshes.at(prim);
        Release(mesh);
        m_byPointsRow.erase(mesh.pointsRow);
        m_byMeshRow.erase(mesh.meshRow);
        m_meshes.erase(prim);
    };

    std::vector<uint32_t> placed;
    if (changes.resync) {
        while (!m_meshes.empty()) {
            forget(m_meshes.begin()->first);
        }
        Prune();
        for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
            if (isMesh(prim)) {
                Place(prim);
                placed.push_back(prim);
            }
        }
    } else {
        for (const StructureEntry& entry : changes.removed) {
            const auto found = m_meshes.find(entry.prim);
            if (found != m_meshes.end() && found->second.generation == entry.generation) {
                forget(entry.prim);
            }
        }
        for (const StructureEntry& entry : changes.added) {
            if (isMesh(entry.prim) && prims.Generation(entry.prim) == entry.generation) {
                if (m_meshes.contains(entry.prim)) {
                    forget(entry.prim);
                }
                Place(entry.prim);
                placed.push_back(entry.prim);
            }
        }
        ERNIE_ZONE("Geometry/Changes");
        std::ranges::sort(placed);
        const auto collect = [&](ColumnRef column, const auto& owners, std::vector<uint32_t>& out) {
            for (const uint32_t row : changes.ArrayRows(column)) {
                const auto owner = owners.find(row);
                if (owner != owners.end() && !std::ranges::binary_search(placed, owner->second)) {
                    out.push_back(owner->second);
                }
            }
            std::ranges::sort(out);
            out.erase(std::ranges::unique(out).begin(), out.end());
        };
        std::vector<uint32_t> moved;
        std::vector<uint32_t> reshaped;
        collect(kPointBasedPoints, m_byPointsRow, moved);
        collect(kMeshFaceVertexIndices, m_byMeshRow, reshaped);
        collect(kMeshFaceVertexCounts, m_byMeshRow, reshaped);
        const Store& store = m_catalog.GetStore();
        std::vector<uint32_t> rekeyed;
        for (const uint32_t prim : moved) {
            if (std::ranges::binary_search(reshaped, prim)) {
                continue;
            }
            Mesh& mesh = m_meshes.at(prim);
            const ArrayRef points = store.Array(kPointBasedPoints, mesh.pointsRow);
            if (mesh.shape && points.address == mesh.key->points &&
                points.count == mesh.key->pointCount) {
                mesh.shape->refit = true;
            } else {
                Release(mesh);
                rekeyed.push_back(prim);
            }
        }
        for (const uint32_t prim : reshaped) {
            Release(m_meshes.at(prim));
        }
        Prune();
        for (const uint32_t prim : reshaped) {
            Place(prim);
        }
        for (const uint32_t prim : rekeyed) {
            Place(prim);
        }
    }
    Settle();
    Prune();
    std::vector<uint32_t> updated;
    Update(updated);
    m_records.Reserve(m_catalog.GetStore().Schema(SchemaId::Mesh).Capacity());
    const bool replaced = m_records.Upload(m_catalog.GetStore().GetDevice());
    if (replaced) {
        m_relinked = {kGlobalWorld};
    }
    if (!m_relinked.empty()) {
        log.Wrote(TableId::Meshes, m_relinked);
    }
    if (!updated.empty()) {
        log.Wrote(TableId::MeshData, updated);
    }
    m_relinked.clear();
}

void Geometry::Update(std::vector<uint32_t>& updated) {
    std::vector<Ref<bvh::Blas>> built;
    std::vector<Ref<bvh::Blas>> refit;
    std::vector<NormalsShape> normals;
    uint32_t total = 0;
    for (auto& [key, shape] : m_shapes) {
        if (!shape.build && !shape.refit) {
            continue;
        }
        shape.updated = Runs();
        (shape.build ? built : refit).push_back(shape.blas);
        const Topology& topology = m_topologies.at(key.topology);
        NormalsShape entry{};
        entry.points.address = key.points;
        entry.triangles.address = topology.triangles.address;
        entry.starts.address = topology.starts.address;
        entry.adjacency.address = topology.adjacency.address;
        entry.normals.address = shape.normals.address;
        entry.bounds.address = shape.bounds.address;
        entry.first = total;
        entry.count = shape.pointCount;
        entry.covered = topology.covered;
        normals.push_back(entry);
        total += shape.pointCount;
        shape.build = false;
        shape.refit = false;
    }
    m_stats.built = uint32_t(built.size());
    m_stats.refit = uint32_t(refit.size());
    m_stats.meshes = uint32_t(m_meshes.size());
    m_stats.shapes = uint32_t(m_shapes.size());
    m_stats.topologies = uint32_t(m_topologies.size());
    if (normals.empty()) {
        return;
    }
    const Prims& prims = m_catalog.GetPrims();
    for (const auto& [prim, mesh] : m_meshes) {
        if (mesh.shape && mesh.shape->updated == Runs() && prims.Alive(prim)) {
            AddWorld(updated, prims.At(prim).world);
        }
    }

    static const profiler::ZoneId blasZone = profiler::ZoneId::Named("Geometry/Blas");
    static const profiler::ZoneId normalsZone = profiler::ZoneId::Named("Geometry/Normals");
    Store& store = m_catalog.GetStore();
    device::Device& device = store.GetDevice();
    store.Flush();
    const Ref<device::Buffer> table =
        device.CreateBuffer({.label = "normals shapes"}, std::span<const NormalsShape>(normals));
    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.BeginZone(blasZone);
    m_builder.Build(encoder, built);
    m_builder.Refit(encoder, refit);
    encoder.EndZone();
    encoder.BeginZone(normalsZone);
    const uint64_t extentWords = uint64_t(normals.size()) * 3;
    const Ref<device::Buffer> extents = device.CreateBuffer(
        {.bytes = extentWords * 2 * sizeof(uint32_t), .label = "shape extents"});
    encoder.Fill(extents->Slice(0, extentWords * sizeof(uint32_t)), ~0u);
    encoder.Fill(extents->Slice(extentWords * sizeof(uint32_t)), 0u);
    NormalsParams params{};
    params.shapes.address = table->Address();
    params.extents.address = extents->Address();
    params.shapeCount = uint32_t(normals.size());
    params.total = total;
    encoder.Dispatch(Kernels().Entry("SmoothNormals"), params,
                     {.x = device::GroupCount(total, kNormalsGroup)});
    encoder.Dispatch(Kernels().Entry("FinishShapeBounds"), params,
                     {.x = device::GroupCount(params.shapeCount, kNormalsGroup)});
    encoder.EndZone();
    queue.Submit(encoder.Finish());
}

void Geometry::Hold(std::vector<Ref<const RefCounted>>& held) const {
    HoldBuffers(held, {m_records.Buffer()});
    for (const auto& [key, shape] : m_shapes) {
        if (shape.blas) {
            held.emplace_back(shape.blas);
        }
    }
}

void Geometry::Prepare() {
    Kernels();
    if (!m_scan) {
        m_scanCapacity = kPreparedScan;
        m_scan =
            std::make_unique<algorithms::Scan>(m_catalog.GetStore().GetDevice(), m_scanCapacity);
    }
}

device::Program& Geometry::Kernels() {
    if (!m_program) {
        m_program = m_catalog.GetStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.kernels.geometry"},
             .library = &kSceneKernels,
             .entries = {"FaceSizes", "FaceTriangles", "TopologySizes", "EmitTriangles",
                         "WriteAdjacency", "WriteStarts", "SmoothNormals", "FinishShapeBounds"},
             .label = "geometry"});
    }
    return *m_program;
}

}

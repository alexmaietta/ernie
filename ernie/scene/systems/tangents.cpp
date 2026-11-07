// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/systems/tangents.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <map>
#include <set>
#include <vector>

#include <mikktspace.h>

#include "ernie/scene/catalog/packing.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/kernels/tables/render_tables.h"
#include "ernie/scene/kernels/tangents.h"
#include "ernie/scene/timeline/changes.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {
namespace {

struct Faces {
    const TangentMesh* mesh = nullptr;
    std::vector<std::array<uint32_t, 4>> corners;
    std::vector<uint32_t> sizes;
    std::vector<Float4>* tangents = nullptr;
};

Faces& ContextFaces(const SMikkTSpaceContext* context) {
    return *static_cast<Faces*>(context->m_pUserData);
}

uint32_t Corner(const SMikkTSpaceContext* context, int face, int vertex) {
    return ContextFaces(context).corners[face][vertex];
}

int FaceCount(const SMikkTSpaceContext* context) {
    return int(ContextFaces(context).corners.size());
}

int FaceSize(const SMikkTSpaceContext* context, int face) {
    return int(ContextFaces(context).sizes[face]);
}

void Position(const SMikkTSpaceContext* context, float out[], int face, int vertex) {
    const Faces& faces = ContextFaces(context);
    const Float3& p = faces.mesh->points[faces.mesh->indices[Corner(context, face, vertex)]];
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
}

void Normal(const SMikkTSpaceContext* context, float out[], int face, int vertex) {
    const Float3& n = ContextFaces(context).mesh->normals[Corner(context, face, vertex)];
    out[0] = n.x;
    out[1] = n.y;
    out[2] = n.z;
}

void TexCoord(const SMikkTSpaceContext* context, float out[], int face, int vertex) {
    const Float2& st = ContextFaces(context).mesh->st[Corner(context, face, vertex)];
    out[0] = st.x;
    out[1] = st.y;
}

void StoreTangent(const SMikkTSpaceContext* context, const float tangent[], float sign, int face,
                  int vertex) {
    (*ContextFaces(context).tangents)[Corner(context, face, vertex)] = {tangent[0], tangent[1],
                                                                        tangent[2], sign};
}

Float3 Sub(Float3 a, Float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Float3 Cross(Float3 a, Float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

}

std::vector<Float3> SmoothCornerNormals(std::span<const Float3> points,
                                        std::span<const int32_t> indices,
                                        std::span<const int32_t> counts) {
    std::vector<Float3> sums(points.size());
    uint32_t first = 0;
    for (const int32_t count : counts) {
        for (int32_t k = 1; k + 1 < count && first + k + 1 < indices.size(); ++k) {
            const uint32_t a = uint32_t(indices[first]);
            const uint32_t b = uint32_t(indices[first + k]);
            const uint32_t c = uint32_t(indices[first + k + 1]);
            if (a >= points.size() || b >= points.size() || c >= points.size()) {
                continue;
            }
            const Float3 n = Cross(Sub(points[b], points[a]), Sub(points[c], points[a]));
            for (const uint32_t point : {a, b, c}) {
                sums[point] = {sums[point].x + n.x, sums[point].y + n.y, sums[point].z + n.z};
            }
        }
        first += uint32_t(std::max(count, 0));
    }
    std::vector<Float3> corners(indices.size());
    for (uint32_t corner = 0; corner < indices.size(); ++corner) {
        const uint32_t point = uint32_t(indices[corner]);
        const Float3 n = point < sums.size() ? sums[point] : Float3{0, 0, 1};
        const float length = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        corners[corner] =
            length > 0 ? Float3{n.x / length, n.y / length, n.z / length} : Float3{0, 0, 1};
    }
    return corners;
}

std::vector<Float4> MikkTangents(const TangentMesh& mesh) {
    std::vector<Float4> tangents(mesh.indices.size(), Float4{1, 0, 0, 1});
    Faces faces{.mesh = &mesh, .tangents = &tangents};
    uint32_t first = 0;
    for (const int32_t count : mesh.counts) {
        if (count == 3 || count == 4) {
            faces.corners.push_back({first, first + 1, first + 2, first + 3});
            faces.sizes.push_back(uint32_t(count));
        } else {
            for (int32_t k = 1; k + 1 < count; ++k) {
                faces.corners.push_back({first, first + k, first + k + 1, 0});
                faces.sizes.push_back(3);
            }
        }
        first += uint32_t(std::max(count, 0));
    }
    if (first > mesh.indices.size() || mesh.normals.size() != mesh.indices.size() ||
        mesh.st.size() != mesh.indices.size()) {
        return {};
    }
    SMikkTSpaceInterface callbacks{};
    callbacks.m_getNumFaces = FaceCount;
    callbacks.m_getNumVerticesOfFace = FaceSize;
    callbacks.m_getPosition = Position;
    callbacks.m_getNormal = Normal;
    callbacks.m_getTexCoord = TexCoord;
    callbacks.m_setTSpaceBasic = StoreTangent;
    SMikkTSpaceContext context{&callbacks, &faces};
    genTangSpaceDefault(&context);
    return tangents;
}

namespace {

constexpr ColumnRef kInputs[] = {
    kPointBasedPoints,  kMeshFaceVertexIndices,          kMeshFaceVertexCounts, kGprimPrimvars,
    kPointBasedNormals, kPointBasedNormalsInterpolation, kMeshSubdivisionScheme};

}

std::span<const ColumnRef> Tangents::Inputs() const {
    return kInputs;
}

void Tangents::Describe(RenderTables& tables) const {
    tables.tangents.address = m_records.Buffer() ? m_records.Buffer()->Address() : 0;
}

void Tangents::Hold(std::vector<Ref<const RefCounted>>& held) const {
    if (m_records.Buffer()) {
        held.emplace_back(m_records.Buffer());
    }
}

void Tangents::CacheCells(Mesh& mesh, const Cells& cells) const {
    mesh.interpolation = cells.interpolation[mesh.pointsRow];
    mesh.smooth = cells.scheme[mesh.meshRow] != uint32_t(SubdivisionScheme::None) ? 1 : 0;
}

Tangents::Key Tangents::KeyFor(const Mesh& mesh) const {
    Store& store = m_catalog.GetStore();
    Key key;
    key.points = store.Array(kPointBasedPoints, mesh.pointsRow).address;
    key.indices = store.Array(kMeshFaceVertexIndices, mesh.meshRow).address;
    key.counts = store.Array(kMeshFaceVertexCounts, mesh.meshRow).address;
    key.primvars = store.Array(kGprimPrimvars, mesh.gprimRow).address;
    key.normals = store.Array(kPointBasedNormals, mesh.pointsRow).address;
    key.interpolation = mesh.interpolation;
    key.smooth = mesh.smooth;
    return key;
}

void Tangents::Compute(const Mesh& mesh, const Key& key, Set& set) {
    ERNIE_ZONE("Tangents/Compute");
    ++m_stats.computed;
    Store& store = m_catalog.GetStore();
    store.Arrays(ColumnType::IntArray).Free(set.tangents);
    set.tangents = {};
    set.words = 0;
    set.deform.reset();
    const ArrayRef points = store.Array(kPointBasedPoints, mesh.pointsRow);
    const ArrayRef indices = store.Array(kMeshFaceVertexIndices, mesh.meshRow);
    const ArrayRef counts = store.Array(kMeshFaceVertexCounts, mesh.meshRow);
    const ArrayRef table = store.Array(kGprimPrimvars, mesh.gprimRow);
    if (points.count == 0 || indices.count == 0 || counts.count == 0 || table.count == 0) {
        return;
    }
    ArrayHeap& ints = store.Arrays(ColumnType::IntArray);
    ArrayHeap& float3s = store.Arrays(ColumnType::Float3Array);
    const std::vector<Primvar> primvars =
        UnpackPrimvars(m_catalog.GetTokens(), Items<int32_t>(ints.Read(table)));
    const auto st = std::ranges::find_if(primvars, [](const Primvar& primvar) {
        return primvar.name == "st" && primvar.type == PrimvarType::Float2;
    });
    if (st == primvars.end()) {
        return;
    }
    TangentMesh input;
    input.points = Items<Float3>(float3s.Read(points));
    input.indices = Items<int32_t>(ints.Read(indices));
    input.counts = Items<int32_t>(ints.Read(counts));
    const std::optional<std::vector<Float2>> corners = CornerValues<Float2>(
        st->Values<Float2>(), st->indices, st->interpolation, input.indices, input.counts);
    if (!corners) {
        return;
    }
    input.st = *corners;
    const ArrayRef normals = store.Array(kPointBasedNormals, mesh.pointsRow);
    std::optional<std::vector<Float3>> authored;
    if (key.smooth == 0 && normals.count > 0) {
        authored =
            CornerValues<Float3>(Items<Float3>(float3s.Read(normals)), {},
                                 Interpolation(key.interpolation), input.indices, input.counts);
    }
    input.normals =
        authored ? *authored : SmoothCornerNormals(input.points, input.indices, input.counts);
    const std::vector<Float4> tangents = MikkTangents(input);
    if (tangents.empty()) {
        return;
    }
    device::Device& device = store.GetDevice();
    ArrayHeap& words = store.Arrays(ColumnType::IntArray);
    set.words = uint32_t(tangents.size() * 4);
    set.tangents = words.Allocate(set.words);
    words.Write(set.tangents, std::as_bytes(std::span(tangents)));

    std::vector<uint32_t> triangles;
    uint32_t first = 0;
    for (const int32_t count : input.counts) {
        for (int32_t k = 1; k + 1 < count; ++k) {
            triangles.insert(triangles.end(),
                             {first, first + uint32_t(k), first + uint32_t(k) + 1});
        }
        first += uint32_t(std::max(count, 0));
    }
    std::map<std::array<uint32_t, 6>, uint32_t> keys;
    std::vector<uint32_t> groups(input.indices.size(), kNoRow);
    for (const uint32_t corner : triangles) {
        const Float2 uv = input.st[corner];
        const Float3 normal = authored ? input.normals[corner] : Float3{};
        const std::array<uint32_t, 6> weld = {
            uint32_t(input.indices[corner]),   std::bit_cast<uint32_t>(uv.x),
            std::bit_cast<uint32_t>(uv.y),     std::bit_cast<uint32_t>(normal.x),
            std::bit_cast<uint32_t>(normal.y), std::bit_cast<uint32_t>(normal.z)};
        groups[corner] = keys.try_emplace(weld, uint32_t(keys.size())).first->second;
    }
    std::vector<uint32_t> starts(keys.size() + 1, 0);
    for (const uint32_t corner : triangles) {
        ++starts[groups[corner] + 1];
    }
    for (uint32_t group = 0; group < keys.size(); ++group) {
        starts[group + 1] += starts[group];
    }
    std::vector<uint32_t> entries(triangles.size());
    std::vector<uint32_t> filled(starts.begin(), starts.end() - 1);
    for (uint32_t entry = 0; entry < triangles.size(); ++entry) {
        entries[filled[groups[triangles[entry]]]++] = entry;
    }
    if (keys.empty()) {
        return;
    }
    Deform deform;
    deform.st = device.CreateBuffer({.label = "tangent st"}, input.st);
    if (authored) {
        deform.normals = device.CreateBuffer({.label = "tangent normals"}, input.normals);
    }
    deform.triangles = device.CreateBuffer({.label = "tangent triangles"}, triangles);
    deform.starts = device.CreateBuffer({.label = "tangent group starts"}, starts);
    deform.corners = device.CreateBuffer({.label = "tangent group corners"}, entries);
    deform.groups = device.CreateBuffer({.label = "tangent corner groups"}, groups);
    deform.results =
        device.CreateBuffer({.bytes = keys.size() * sizeof(Float4), .label = "tangent groups"});
    deform.groupCount = uint32_t(keys.size());
    deform.cornerCount = uint32_t(input.indices.size());
    set.deform = std::move(deform);
}

void Tangents::Recompute(const Mesh& mesh, const Key& key, Set& set,
                         device::CommandEncoder& encoder) {
    ++m_stats.deformed;
    const Deform& deform = *set.deform;
    TangentParams params{};
    params.points.address = key.points;
    params.smoothNormals.address = m_geometry.Record(mesh.meshRow).normals.address;
    params.cornerNormals.address = deform.normals ? deform.normals->Address() : 0;
    params.cornerSt.address = deform.st->Address();
    params.indices.address = key.indices;
    params.triangles.address = deform.triangles->Address();
    params.groupStarts.address = deform.starts->Address();
    params.groupCorners.address = deform.corners->Address();
    params.cornerGroups.address = deform.groups->Address();
    params.groupTangents.address = deform.results->Address();
    params.tangents.address = set.tangents.address;
    params.groupCount = deform.groupCount;
    params.cornerCount = deform.cornerCount;
    encoder.Dispatch(Kernels().Entry("GroupTangents"), params,
                     {.x = device::GroupCount(params.groupCount, kTangentsGroup)});
    encoder.Dispatch(Kernels().Entry("CornerTangents"), params,
                     {.x = device::GroupCount(params.cornerCount, kTangentsGroup)});
}

void Tangents::Link(uint32_t prim, const Mesh& mesh) {
    TangentRecord record{};
    if (mesh.key) {
        const Set& set = m_sets.at(*mesh.key);
        if (set.words > 0) {
            record.tangents = set.tangents;
            record.primvar = m_catalog.GetTokens().Find("st").id;
        }
    }
    const Prims& prims = m_catalog.GetPrims();
    if (m_records.Set(mesh.meshRow, record) && prims.Alive(prim)) {
        const uint32_t world = prims.At(prim).world;
        if (std::ranges::find(m_relinked, world) == m_relinked.end()) {
            m_relinked.push_back(world);
        }
    }
}

void Tangents::Place(uint32_t prim, const Cells& cells) {
    const Prims& prims = m_catalog.GetPrims();
    Mesh mesh;
    mesh.generation = prims.Generation(prim);
    mesh.pointsRow = prims.Row(prim, SchemaId::PointBased);
    mesh.meshRow = prims.Row(prim, SchemaId::Mesh);
    mesh.gprimRow = prims.Row(prim, SchemaId::Gprim);
    m_byPointsRow[mesh.pointsRow] = prim;
    m_byMeshRow[mesh.meshRow] = prim;
    m_byGprimRow[mesh.gprimRow] = prim;
    CacheCells(mesh, cells);
    const Key key = KeyFor(mesh);
    const auto [entry, fresh] = m_sets.try_emplace(key);
    if (fresh) {
        Compute(mesh, key, entry->second);
    }
    ++entry->second.users;
    mesh.key = key;
    mesh.set = &entry->second;
    Link(prim, mesh);
    m_meshes[prim] = mesh;
}

void Tangents::Prune() {
    if (!m_stale) {
        return;
    }
    m_stale = false;
    ArrayHeap& words = m_catalog.GetStore().Arrays(ColumnType::IntArray);
    std::erase_if(m_sets, [&](const auto& entry) {
        if (entry.second.users > 0) {
            return false;
        }
        words.Free(entry.second.tangents);
        return true;
    });
}

void Tangents::Release(Mesh& mesh) {
    if (mesh.key) {
        --m_sets.at(*mesh.key).users;
        mesh.key.reset();
        mesh.set = nullptr;
        m_stale = true;
    }
}

void Tangents::Forget(uint32_t prim) {
    Mesh& mesh = m_meshes.at(prim);
    Release(mesh);
    m_byPointsRow.erase(mesh.pointsRow);
    m_byMeshRow.erase(mesh.meshRow);
    m_byGprimRow.erase(mesh.gprimRow);
    m_meshes.erase(prim);
}

device::Operation Tangents::Run(const ChangeSet& changes, Stream) {
    const device::Operation done = Process(changes);
    m_records.Reserve(m_catalog.GetStore().Schema(SchemaId::Mesh).Capacity());
    if (m_records.Upload(m_catalog.GetStore().GetDevice())) {
        m_relinked = {kGlobalWorld};
    }
    if (changes.log != nullptr && !m_relinked.empty()) {
        changes.log->Wrote(TableId::Tangents, m_relinked);
    }
    m_relinked.clear();
    return done;
}

device::Operation Tangents::Process(const ChangeSet& changes) {
    ERNIE_ZONE("Tangents/Run");
    ++m_runs;
    m_stats.computed = 0;
    Store& store = m_catalog.GetStore();
    const Prims& prims = m_catalog.GetPrims();
    const bool rekey = changes.resync ||
                       store.LastWrite(kPointBasedNormalsInterpolation) > m_seen ||
                       store.LastWrite(kMeshSubdivisionScheme) > m_seen;
    m_seen = store.WriteSequence();
    m_stats.deformed = 0;
    bool read = false;
    const auto cells = [&]() -> const Cells& {
        Table& points = store.Schema(SchemaId::PointBased);
        Table& meshes = store.Schema(SchemaId::Mesh);
        const bool stale = rekey || changes.resync || !changes.added.empty() ||
                           m_cells.interpolation.size() < points.Rows() ||
                           m_cells.scheme.size() < meshes.Rows();
        if (!read && stale) {
            m_cells.interpolation = Items<uint32_t>(
                points.Read(kPointBasedNormalsInterpolation.index, 0, points.Rows()));
            m_cells.scheme =
                Items<uint32_t>(meshes.Read(kMeshSubdivisionScheme.index, 0, meshes.Rows()));
            read = true;
        }
        return m_cells;
    };
    const auto isMesh = [&](uint32_t prim) {
        return prims.Alive(prim) && IsA(prims.At(prim).schema, SchemaId::Mesh);
    };

    std::set<uint32_t> placed;
    if (changes.resync) {
        while (!m_meshes.empty()) {
            Forget(m_meshes.begin()->first);
        }
        for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
            if (isMesh(prim)) {
                placed.insert(prim);
            }
        }
    } else {
        for (const StructureEntry& entry : changes.removed) {
            const auto found = m_meshes.find(entry.prim);
            if (found != m_meshes.end() && found->second.generation == entry.generation) {
                Forget(entry.prim);
            }
        }
        for (const StructureEntry& entry : changes.added) {
            if (isMesh(entry.prim) && prims.Generation(entry.prim) == entry.generation) {
                if (m_meshes.contains(entry.prim)) {
                    Forget(entry.prim);
                }
                placed.insert(entry.prim);
            }
        }
    }

    std::set<uint32_t> affected;
    std::set<uint32_t> reshaped;
    if (rekey) {
        for (auto& [prim, mesh] : m_meshes) {
            CacheCells(mesh, cells());
            affected.insert(prim);
            reshaped.insert(prim);
        }
    }
    const auto collect = [&](ColumnRef column, const auto& owners, bool points) {
        for (const uint32_t row : changes.ArrayRows(column)) {
            if (const auto owner = owners.find(row); owner != owners.end()) {
                if (points) {
                    const Mesh& mesh = m_meshes.at(owner->second);
                    if (mesh.set && mesh.set->words == 0 &&
                        store.Array(kPointBasedPoints, mesh.pointsRow).address ==
                            mesh.key->points) {
                        continue;
                    }
                }
                affected.insert(owner->second);
                if (!points) {
                    reshaped.insert(owner->second);
                }
            }
        }
    };
    collect(kPointBasedPoints, m_byPointsRow, true);
    collect(kPointBasedNormals, m_byPointsRow, false);
    collect(kMeshFaceVertexIndices, m_byMeshRow, false);
    collect(kMeshFaceVertexCounts, m_byMeshRow, false);
    collect(kGprimPrimvars, m_byGprimRow, false);

    std::set<Key> stale;
    std::set<Key> exact;
    std::vector<uint32_t> rekeyed;
    for (const uint32_t prim : affected) {
        if (placed.contains(prim) || !m_meshes.contains(prim)) {
            continue;
        }
        Mesh& mesh = m_meshes.at(prim);
        const Key key = KeyFor(mesh);
        if (mesh.key && *mesh.key == key) {
            stale.insert(key);
            if (reshaped.contains(prim)) {
                exact.insert(key);
            }
            continue;
        }
        Forget(prim);
        rekeyed.push_back(prim);
    }
    Prune();
    for (const uint32_t prim : placed) {
        Place(prim, cells());
    }
    for (const uint32_t prim : rekeyed) {
        Place(prim, cells());
    }
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Tangents/Recompute");
    device::Device& device = store.GetDevice();
    std::optional<device::CommandEncoder> encoder;
    std::optional<device::ZoneScope> scope;
    std::map<Key, std::vector<uint32_t>> users;
    if (!stale.empty()) {
        for (const auto& [prim, mesh] : m_meshes) {
            if (mesh.key && stale.contains(*mesh.key)) {
                users[*mesh.key].push_back(prim);
            }
        }
    }
    for (const auto& [key, owners] : users) {
        Set& set = m_sets.at(key);
        bool computed = false;
        for (const uint32_t prim : owners) {
            const Mesh& mesh = m_meshes.at(prim);
            if (!computed) {
                const bool onDevice =
                    set.deform && !exact.contains(key) &&
                    (set.deform->normals || m_geometry.Record(mesh.meshRow).normals.address != 0);
                if (onDevice) {
                    if (!encoder) {
                        encoder = device.GetQueue().CreateEncoder();
                        scope.emplace(*encoder, zone);
                    }
                    Recompute(mesh, key, set, *encoder);
                } else {
                    Compute(mesh, key, set);
                }
                computed = true;
            }
            Link(prim, mesh);
        }
    }
    Prune();
    m_stats.meshes = uint32_t(m_meshes.size());
    m_stats.sets = uint32_t(m_sets.size());
    if (encoder) {
        scope.reset();
        store.Flush();
        return device.GetQueue().Submit(encoder->Finish());
    }
    return {};
}

void Tangents::Prepare() {
    Kernels();
}

device::Program& Tangents::Kernels() {
    if (!m_program) {
        m_program = m_catalog.GetStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.kernels.tangents"},
             .library = &kSceneKernels,
             .entries = {"GroupTangents", "CornerTangents"},
             .label = "tangents"});
    }
    return *m_program;
}

}

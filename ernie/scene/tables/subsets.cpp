// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/subsets.h"

#include <algorithm>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/kernels/subsets.h"
#include "ernie/stage/publish/change_set.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {

namespace {

constexpr ColumnRef kInputs[] = {kGeomSubsetElementType, kGeomSubsetIndices, kGeomSubsetFamilyName,
                                 kMaterialBindingAPIMaterialBinding, kMeshFaceVertexCounts};

}

std::span<const ColumnRef> SubsetTable::Inputs() const {
    return kInputs;
}

void SubsetTable::Describe(RenderTables& tables) const {
    tables.subsets.address = m_records.Buffer() ? m_records.Buffer()->Address() : 0;
}

void SubsetTable::Run(const ChangeSet& changes, TableVersions&) {
    ERNIE_ZONE("Subsets/Run");
    m_stats.rebuilt = false;
    const Prims& prims = m_structure.GetPrims();
    ColumnStore& store = m_structure.GetColumnStore();
    const auto isSubset = [&](uint32_t prim) {
        return prims.Alive(prim) && prims.Record(prim).schema == SchemaId::GeomSubset;
    };
    bool dirty = !m_built || changes.resync;
    if (dirty) {
        m_subsets.clear();
        for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
            if (isSubset(prim)) {
                m_subsets.insert(prim);
            }
        }
    } else {
        for (const StructureEntry& entry : changes.removed) {
            dirty = m_subsets.erase(entry.prim) > 0 || dirty;
        }
        for (const StructureEntry& entry : changes.added) {
            if (isSubset(entry.prim) && prims.Generation(entry.prim) == entry.generation) {
                m_subsets.insert(entry.prim);
                dirty = true;
            }
        }
        for (const StructureEntry& entry : changes.apis) {
            dirty = dirty ||
                    (entry.api == SchemaId::SemanticsLabelsAPI && m_subsets.contains(entry.prim));
        }
    }
    dirty = dirty || (!m_subsets.empty() && std::ranges::any_of(kInputs, [&](ColumnRef column) {
                return store.LastWrite(column) > m_seen;
            }));
    m_seen = store.WriteSequence();
    m_built = true;
    if (dirty) {
        Rebuild();
    }
    m_records.Reserve(store.Schema(SchemaId::Mesh).Capacity());
    m_records.Upload(store.GetDevice());
}

void SubsetTable::Rebuild() {
    ERNIE_ZONE("Subsets/Rebuild");
    m_stats = {.rebuilt = true};
    const Prims& prims = m_structure.GetPrims();
    const Paths& paths = m_structure.GetPaths();
    ColumnStore& store = m_structure.GetColumnStore();
    ArrayPool& ints = store.Arrays(ColumnType::IntArray);
    for (const auto& [meshRow, record] : m_written) {
        ints.Free(record.materials);
        ints.Free(record.parts);
        m_records.Set(meshRow, {});
    }
    m_written.clear();
    if (m_subsets.empty()) {
        return;
    }

    ColumnTable& table = store.Schema(SchemaId::GeomSubset);
    const std::vector<uint32_t> elements =
        Items<uint32_t>(table.Read(kGeomSubsetElementType.index, 0, table.Rows()));
    const std::vector<uint32_t> families =
        Items<uint32_t>(table.Read(kGeomSubsetFamilyName.index, 0, table.Rows()));
    const uint32_t materialBind = m_structure.GetTokens().Intern("materialBind").id;
    std::set<uint32_t> meshes;
    for (const uint32_t subset : m_subsets) {
        const uint32_t parent = paths.Parent(subset);
        if (parent != kNoPrim && prims.Alive(parent) &&
            IsA(prims.Record(parent).schema, SchemaId::Mesh)) {
            meshes.insert(parent);
        }
    }

    std::vector<SubsetJob> jobs;
    std::vector<SubsetOutput> outputs;
    uint32_t indexCount = 0;
    uint32_t entryCount = 0;
    for (const uint32_t mesh : meshes) {
        const uint32_t meshRow = prims.Row(mesh, SchemaId::Mesh);
        const uint32_t faces = store.Array(kMeshFaceVertexCounts, meshRow).count;
        if (faces == 0) {
            continue;
        }
        const std::vector<uint32_t> children = paths.Children(mesh);
        SubsetRecord record{};
        for (const SubsetMap map : {SubsetMap::Materials, SubsetMap::Parts}) {
            const size_t firstJob = jobs.size();
            for (const uint32_t child : children) {
                if (!m_subsets.contains(child)) {
                    continue;
                }
                const uint32_t row = prims.Row(child, SchemaId::GeomSubset);
                const ArrayRef indices = store.Array(kGeomSubsetIndices, row);
                const bool wanted = map == SubsetMap::Materials
                                        ? families[row] == materialBind
                                        : prims.HasApi(child, SchemaId::SemanticsLabelsAPI);
                if (!wanted || elements[row] != uint32_t(ElementType::Face) || indices.count == 0) {
                    continue;
                }
                SubsetJob job{};
                job.indices.address = indices.address;
                job.subset = child;
                job.count = indices.count;
                job.firstIndex = indexCount;
                job.firstEntry = entryCount;
                job.faceCount = faces;
                job.map = uint32_t(map);
                jobs.push_back(job);
                indexCount += indices.count;
                ++m_stats.subsets;
            }
            if (jobs.size() == firstJob) {
                continue;
            }
            const ArrayRef values = ints.Allocate(faces);
            (map == SubsetMap::Materials ? record.materials : record.parts) = values;
            SubsetOutput output{};
            output.values.address = values.address;
            output.firstEntry = entryCount;
            output.count = faces;
            outputs.push_back(output);
            entryCount += faces;
        }
        if (record.materials.count > 0 || record.parts.count > 0) {
            m_written[meshRow] = record;
            m_records.Set(meshRow, record);
            ++m_stats.meshes;
        }
    }
    if (jobs.empty()) {
        return;
    }

    store.Flush();
    device::Device& device = store.GetDevice();
    const Ref<device::Buffer> jobBuffer =
        device.CreateBuffer({.label = "subset jobs"}, std::span<const SubsetJob>(jobs));
    const Ref<device::Buffer> outputBuffer =
        device.CreateBuffer({.label = "subset outputs"}, std::span<const SubsetOutput>(outputs));
    const Ref<device::Buffer> ranks =
        device.CreateBuffer({.bytes = uint64_t(entryCount) * 4, .label = "subset face ranks"});
    SubsetParams params{};
    params.scene.address = store.DeviceData();
    params.jobs.address = jobBuffer->Address();
    params.outputs.address = outputBuffer->Address();
    params.ranks.address = ranks->Address();
    params.jobCount = uint32_t(jobs.size());
    params.outputCount = uint32_t(outputs.size());
    params.indexCount = indexCount;
    params.entryCount = entryCount;
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Subsets/Write");
    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    {
        const device::ZoneScope scope(encoder, zone);
        encoder.Fill(ranks->Slice(0), ~0u);
        encoder.Dispatch(Kernels().Entry("ClaimFaces"), params,
                         {.x = device::GroupCount(indexCount, kSubsetGroup)});
        encoder.Dispatch(Kernels().Entry("WriteFaces"), params,
                         {.x = device::GroupCount(entryCount, kSubsetGroup)});
    }
    queue.Submit(encoder.Finish());
}

void SubsetTable::Prepare() {
    Kernels();
}

device::Program& SubsetTable::Kernels() {
    if (!m_program) {
        m_program = m_structure.GetColumnStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.tables.kernels.subsets"},
             .library = &kSceneKernels,
             .entries = {"ClaimFaces", "WriteFaces"},
             .label = "subsets"});
    }
    return *m_program;
}

}

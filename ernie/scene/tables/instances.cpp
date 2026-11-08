// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/instances.h"

#include <algorithm>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/kernels/instances.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

constexpr ColumnRef kActivation[] = {kSphereRadius,
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
                                     kSphereLightRadius,
                                     kSphereLightTreatAsPoint,
                                     kRectLightWidth,
                                     kRectLightHeight,
                                     kDiskLightRadius,
                                     kCylinderLightLength,
                                     kCylinderLightRadius,
                                     kCylinderLightTreatAsLine,
                                     kLightAPIVisibleInPrimaryRay,
                                     kPointInstancerProtoIndices,
                                     kPointInstancerIds,
                                     kPointInstancerInvisibleIds};

constexpr ColumnRef kPlacement[] = {kPointBasedPoints, kPointInstancerOrientations,
                                    kPointInstancerScales, kRenderProductCamera};

constexpr ColumnRef kLayoutInputs[] = {kPointInstancerPrototypes, kPointInstancerPositions,
                                       kPrototypeAPIPrototype};

const std::vector<ColumnRef>& InputColumns() {
    static const std::vector<ColumnRef> inputs = [] {
        std::vector<ColumnRef> columns(std::begin(kPlacement), std::end(kPlacement));
        columns.insert(columns.end(), std::begin(kActivation), std::end(kActivation));
        columns.insert(columns.end(), std::begin(kLayoutInputs), std::end(kLayoutInputs));
        return columns;
    }();
    return inputs;
}

}

InstanceTable::InstanceTable(Structure& structure, bvh::Builder& builder, uint32_t maxRefits)
    : m_structure(structure), m_builder(builder), m_maxRefits(maxRefits) {
    device::Device& device = structure.GetColumnStore().GetDevice();
    const BvhBox unit = {{-1, -1, -1}, {1, 1, 1}};
    m_unitBoxes = device.CreateBuffer(
        {.usage = device::BufferUsage::AccelBuildInput, .label = "unit box"}, std::span(&unit, 1));
    m_unitBox = builder.CreateBlas(device::AccelBoxes{.boxes = m_unitBoxes->Slice(0), .count = 1});
    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    const Ref<bvh::Blas> boxes[] = {m_unitBox};
    builder.Build(encoder, boxes);
    queue.Submit(encoder.Finish());
}

std::span<const ColumnRef> InstanceTable::Inputs() const {
    return InputColumns();
}

void InstanceTable::Run(const ChangeSet& changes, TableVersions& versions) {
    ERNIE_ZONE("Instances/Run");
    ColumnStore& store = m_structure.GetColumnStore();
    const uint64_t seen = store.WriteSequence();
    bool relaid = changes.resync || !m_tlas || m_structure.CompositionVersion() != m_compositions;
    bool restacked = false;
    if (!relaid &&
        (std::ranges::any_of(kLayoutInputs,
                             [&](ColumnRef column) { return store.LastWrite(column) > m_seen; }) ||
         store.LastWrite(kPointInstancerInvisibleIds) > m_seen)) {
        std::vector<Instancer> fresh = GatherInstancers();
        if (fresh != m_instancers) {
            const bool counts =
                std::ranges::equal(fresh, m_instancers, [](const Instancer& a, const Instancer& b) {
                    return a.prim == b.prim && a.world == b.world && a.native == b.native &&
                           a.roots == b.roots;
                });
            m_instancers = std::move(fresh);
            relaid = !counts || !Restack();
            restacked = !relaid;
        } else if (store.LastWrite(kPointInstancerInvisibleIds) > m_seen) {
            relaid = !Restack();
            restacked = !relaid;
        }
    }
    std::set<uint32_t> structural;
    if (!relaid && (changes.structure || !changes.apis.empty())) {
        if (std::optional<std::set<uint32_t>> touched = Apply(changes)) {
            structural = std::move(*touched);
        } else {
            relaid = true;
        }
    }
    if (relaid) {
        Relayout();
    }
    if (relaid || restacked) {
        versions.Record(TableId::Instances, std::vector<uint32_t>{kGlobalWorld});
    }
    const uint32_t tlases = m_regions.Size();
    if (restacked) {
        for (uint32_t tlas = 0; tlas < tlases; ++tlas) {
            structural.insert(tlas);
        }
    }
    std::set<uint32_t> touched = structural;
    const bool global = relaid || store.LastWrite(kRenderProductCamera) > m_seen ||
                        std::ranges::find(changes.worlds, kGlobalWorld) != changes.worlds.end();
    for (const uint32_t world : global ? std::vector<uint32_t>{kGlobalWorld} : changes.worlds) {
        touched.merge(Tlases(world));
    }
    const std::vector<uint32_t> dirty(touched.begin(), touched.end());
    const auto written = [&](std::span<const ColumnRef> columns) {
        return std::ranges::any_of(
            columns, [&](ColumnRef column) { return store.LastWrite(column) > m_seen; });
    };
    const bool activated = changes.Wrote(TableId::Imageables) || changes.Wrote(TableId::Meshes) ||
                           written(kActivation);
    const bool placed = relaid || restacked || activated || !structural.empty() ||
                        changes.Wrote(TableId::Transforms) || changes.Wrote(TableId::MeshPoints) ||
                        written(kPlacement) || written(kLayoutInputs);
    m_seen = seen;

    m_stats = {};
    m_stats.tlases = tlases;
    m_stats.instances = 0;
    for (const uint32_t count : m_regions.Counts()) {
        m_stats.instances += count;
    }
    m_stats.relaid = relaid;
    if (dirty.empty()) {
        return;
    }

    std::vector<uint32_t> built;
    std::vector<uint32_t> refit;
    uint32_t widest = 0;
    for (const uint32_t tlas : dirty) {
        const uint32_t count = m_regions.Counts()[tlas];
        widest = std::max(widest, m_regions.Capacity(tlas));
        m_stats.written += count;
        if (!placed) {
            continue;
        }
        if (relaid || activated || structural.contains(tlas) || m_refits[tlas] >= m_maxRefits) {
            built.push_back(tlas);
            m_refits[tlas] = 0;
        } else {
            refit.push_back(tlas);
            ++m_refits[tlas];
        }
    }
    m_stats.built = uint32_t(built.size());
    if (!built.empty()) {
        std::vector<uint32_t> rebuilt;
        for (const uint32_t tlas : built) {
            rebuilt.push_back(TlasWorld(tlas, m_worlds));
        }
        versions.Record(TableId::Slots, rebuilt);
    }
    m_stats.refit = uint32_t(refit.size());

    device::Device& device = store.GetDevice();
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Instances/Write");
    const Ref<device::Buffer> dirtyBuffer = device.CreateBuffer({.label = "dirty TLASes"}, dirty);
    InstanceParams params{};
    params.scene.address = store.DeviceData();
    params.layout.address = m_layout->Address();
    params.instances.address = m_records.Address();
    params.shapes.address = m_shapes.Address();
    params.tlases.address = dirtyBuffer->Address();
    params.hits.address = m_hits.Address();
    params.origins.address = m_renderOrigins.Address();
    params.transforms = changes.tables->transforms;
    params.imageables = changes.tables->imageables;
    params.meshes = changes.tables->meshes;
    params.slotWorlds.address = m_slotWorlds.Address();
    params.unitBox = m_unitBox->Handle();

    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    OriginParams origins{};
    origins.scene.address = store.DeviceData();
    origins.layout.address = m_layout->Address();
    origins.products.address = m_productBuffer->Address();
    origins.origins.address = m_renderOrigins.Address();
    origins.viewers.address = m_viewers->Address();
    origins.transforms = changes.tables->transforms;
    origins.productCount = uint32_t(m_products.size());
    origins.tlasCount = tlases;
    if (!m_products.empty()) {
        encoder.Dispatch(Kernels().Entry("GatherViewers"), origins,
                         {.x = device::GroupCount(origins.productCount, kInstanceGroup)});
    }
    encoder.Dispatch(Kernels().Entry("PlaceOrigins"), origins,
                     {.x = device::GroupCount(tlases, kInstanceGroup)});
    if (widest > 0) {
        const device::ZoneScope scope(encoder, zone);
        encoder.Dispatch(
            Kernels().Entry("WriteInstances"), params,
            {.x = device::GroupCount(widest, kInstanceGroup), .y = uint32_t(dirty.size())});
    }
    const device::BufferRange counts =
        LayoutWords(m_header.instanceCounts.address, 0, m_regions.Size());
    if (!built.empty()) {
        m_builder.Build(encoder, *m_tlas, *m_records.Buffer(), counts, built);
    }
    if (!refit.empty()) {
        m_builder.Refit(encoder, *m_tlas, *m_records.Buffer(), counts, refit);
    }
    queue.Submit(encoder.Finish());
}

std::set<uint32_t> InstanceTable::Tlases(uint32_t world) const {
    std::set<uint32_t> tlases;
    if (world == kGlobalWorld) {
        for (uint32_t tlas = 0; tlas < m_regions.Size(); ++tlas) {
            tlases.insert(tlas);
        }
        return tlases;
    }
    if (world < m_worlds) {
        tlases.insert(world);
    }
    for (const auto* map : {&m_shownIn, &m_watch}) {
        if (const auto found = map->find(world); found != map->end()) {
            tlases.insert(found->second.begin(), found->second.end());
        }
    }
    return tlases;
}

void InstanceTable::Describe(RenderTables& tables) const {
    tables.instances.address = m_layout ? m_layout->Address() : 0;
    tables.origins.address = m_renderOrigins.Address();
    tables.slotWorlds.address = m_slotWorlds.Address();
}

void InstanceTable::Hold(std::vector<Ref<const RefCounted>>& held) const {
    HoldBuffers(held, {m_layout, m_records.Buffer(), m_shapes.Buffer(), m_hits.Buffer(),
                       m_slotWorlds.Buffer(), m_instancing, m_renderOrigins.Buffer()});
    if (m_tlas) {
        held.emplace_back(m_tlas);
    }
    if (m_unitBox) {
        held.emplace_back(m_unitBox);
    }
}

void InstanceTable::Prepare() {
    Kernels();
}

device::Program& InstanceTable::Kernels() {
    if (!m_program) {
        m_program = m_structure.GetColumnStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.kernels.instances"},
             .library = &kSceneKernels,
             .entries = {"WriteInstances", "GatherViewers", "PlaceOrigins"},
             .label = "instances"});
    }
    return *m_program;
}

}

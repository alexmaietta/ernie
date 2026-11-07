// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/systems/instances.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <numeric>

#include "ernie/device/commands.h"
#include "ernie/device/kernels/shared.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/kernels/instances.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/kernels/tables/render_tables.h"
#include "ernie/scene/timeline/changes.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

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
        std::vector<ColumnRef> columns = {
            kPointBasedPoints,      kPointBasedNormals,     kPointBasedNormalsInterpolation,
            kMeshFaceVertexIndices, kMeshSubdivisionScheme, kPointInstancerOrientations,
            kPointInstancerScales,  kRenderProductCamera,   kGprimPrimvars,
            kGprimDisplayColor,     kPointInstancerPrimvars};
        columns.insert(columns.end(), std::begin(kActivation), std::end(kActivation));
        columns.insert(columns.end(), std::begin(kLayoutInputs), std::end(kLayoutInputs));
        return columns;
    }();
    return inputs;
}

bool Drawable(SchemaId schema) {
    return IsA(schema, SchemaId::Gprim) || IsEmitter(uint32_t(schema));
}

}

Instances::Instances(Catalog& catalog, bvh::Builder& builder, uint32_t maxRefits)
    : m_catalog(catalog), m_builder(builder), m_maxRefits(maxRefits) {
    device::Device& device = catalog.GetStore().GetDevice();
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

std::span<const ColumnRef> Instances::Inputs() const {
    return InputColumns();
}

std::span<const ColumnRef> Instances::Outputs() const {
    return {};
}

bool Instances::ShowsGlobals(uint32_t tlas) const {
    const auto composed = m_composed.find(tlas);
    return composed == m_composed.end() || composed->second.global;
}

std::vector<uint32_t> Instances::RegionItems(uint32_t tlas) const {
    std::vector<uint32_t> items = m_blockItems[tlas];
    if (tlas < m_worlds) {
        items.insert(items.end(), m_own[tlas].begin(), m_own[tlas].end());
    }
    if (const auto composed = m_composed.find(tlas); composed != m_composed.end()) {
        for (const uint32_t world : composed->second.worlds) {
            if (world < m_worlds && world != tlas) {
                items.insert(items.end(), m_own[world].begin(), m_own[world].end());
            }
        }
    }
    if (ShowsGlobals(tlas)) {
        items.insert(items.end(), m_globals.begin(), m_globals.end());
    }
    return items;
}

void Instances::Compose() {
    m_compositions = m_catalog.CompositionVersion();
    m_composed.clear();
    m_shownIn.clear();
    for (const auto& [world, desc] : m_catalog.Compositions()) {
        if (world >= m_worlds) {
            continue;
        }
        m_composed[world] = desc;
        for (const uint32_t shown : desc.worlds) {
            if (shown < m_worlds && shown != world) {
                m_shownIn[shown].insert(world);
            }
        }
    }
}

std::vector<Instances::Instancer> Instances::GatherInstancers() const {
    const Prims& prims = m_catalog.GetPrims();
    Store& store = m_catalog.GetStore();
    store.Flush();
    const auto targets = [&](ColumnRef column, uint32_t row) {
        const ArrayRef array = store.Array(column, row);
        std::vector<uint32_t> found =
            array.address == 0 ? std::vector<uint32_t>{}
                               : Items<uint32_t>(store.Arrays(ColumnType::PrimArray).Read(array));
        for (uint32_t& prim : found) {
            if (prim != kNoPrim && !prims.Alive(prim)) {
                prim = kNoPrim;
            }
        }
        return found;
    };
    std::vector<Instancer> instancers;
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim)) {
            continue;
        }
        const PrimRecord& record = prims.At(prim);
        Instancer instancer{.prim = prim, .world = record.world};
        if (record.schema == SchemaId::PointInstancer) {
            const uint32_t row = prims.Row(prim, SchemaId::PointInstancer);
            instancer.count = store.Array(kPointInstancerPositions, row).count;
            instancer.roots = targets(kPointInstancerPrototypes, row);
        } else if (prims.HasApi(prim, SchemaId::PrototypeAPI)) {
            instancer.native = true;
            instancer.count = 1;
            instancer.roots =
                targets(kPrototypeAPIPrototype, prims.Row(prim, SchemaId::PrototypeAPI));
            instancer.roots.resize(std::min<size_t>(instancer.roots.size(), 1));
        } else {
            continue;
        }
        instancers.push_back(std::move(instancer));
    }
    return instancers;
}

bool Instances::Member(uint32_t prim) const {
    const Prims& prims = m_catalog.GetPrims();
    for (uint32_t at = prim; at != kNoPrim; at = prims.At(at).parent) {
        if (m_roots.contains(at)) {
            return true;
        }
    }
    return false;
}

void Instances::Relayout() {
    ERNIE_ZONE("Instances/Relayout");
    const Prims& prims = m_catalog.GetPrims();
    device::Device& device = m_catalog.GetStore().GetDevice();
    m_worlds = prims.WorldCount();
    m_own.assign(m_worlds, {});
    m_globals.clear();
    m_origins.clear();
    m_products.clear();
    m_instancers = GatherInstancers();
    m_roots.clear();
    for (const Instancer& instancer : m_instancers) {
        for (const uint32_t root : instancer.roots) {
            if (root != kNoPrim) {
                m_roots.insert(root);
            }
        }
    }
    m_members.clear();
    m_drawables.clear();
    m_nested.clear();
    std::set<uint32_t> instancers;
    for (const Instancer& instancer : m_instancers) {
        instancers.insert(instancer.prim);
    }
    Compose();
    std::vector<uint32_t> origins(std::max(m_worlds, 1u), kNoRow);
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim)) {
            continue;
        }
        const PrimRecord& record = prims.At(prim);
        uint32_t root = kNoPrim;
        for (uint32_t at = prim; at != kNoPrim && root == kNoPrim; at = prims.At(at).parent) {
            if (m_roots.contains(at)) {
                root = at;
            }
        }
        if (root != kNoPrim) {
            m_members.insert(prim);
            if (Drawable(record.schema)) {
                m_drawables[root].push_back(prim);
            }
            if (instancers.contains(prim)) {
                m_nested[root].push_back(prim);
            }
        } else if (Drawable(record.schema)) {
            (record.world == kGlobalWorld ? m_globals : m_own[record.world]).push_back(prim);
        }
        if (IsA(record.schema, SchemaId::World)) {
            origins[record.world] = prims.Row(prim, SchemaId::Xformable);
            m_origins.insert(prim);
        }
        if (record.schema == SchemaId::RenderProduct) {
            m_products.push_back(prim);
        }
    }
    m_productBuffer =
        device.CreateBuffer({.label = "render products"},
                            std::span<const uint32_t>(
                                m_products.empty() ? std::vector<uint32_t>{kNoPrim} : m_products));

    BuildInstancing();

    std::vector<std::vector<uint32_t>> lists;
    for (uint32_t tlas = 0; tlas <= m_worlds; ++tlas) {
        lists.push_back(RegionItems(tlas));
    }
    m_regions.Assign(std::move(lists));
    std::vector<uint32_t> capacities;
    for (uint32_t tlas = 0; tlas <= m_worlds; ++tlas) {
        capacities.push_back(m_regions.Capacity(tlas));
        if (capacities.back() > kInstanceCustomMask + 1) {
            throw Error("{} instances in one TLAS; a TLAS holds at most {}", capacities.back(),
                        kInstanceCustomMask + 1);
        }
    }

    m_tlas = m_builder.CreateTlas(capacities);
    m_records = device.CreateBuffer(
        {.bytes = std::max<uint64_t>(m_regions.Items().size(), 1) * sizeof(InstanceRecord),
         .usage = device::BufferUsage::AccelBuildInput,
         .label = "instances"});
    m_surfaces = device.CreateBuffer(
        {.bytes = std::max<uint64_t>(m_regions.Items().size(), 1) * sizeof(SurfaceRecord),
         .label = "instance surfaces"});
    m_shapes = device.CreateBuffer(
        {.bytes = std::max<uint64_t>(m_regions.Items().size(), 1) * sizeof(ShapeRecord),
         .label = "instance shapes"});
    m_hits = device.CreateBuffer(
        {.bytes = std::max<uint64_t>(m_regions.Items().size(), 1) * sizeof(SlotHit),
         .label = "instance hits"});
    device::CommandEncoder clear = device.GetQueue().CreateEncoder();
    clear.Fill(*m_records);
    clear.Fill(*m_shapes);
    clear.Fill(*m_surfaces);
    clear.Fill(*m_hits);
    device.GetQueue().Submit(clear.Finish());

    std::vector<uint32_t> words(sizeof(InstanceLayout) / sizeof(uint32_t));
    const auto append = [&](std::span<const uint32_t> values) {
        const uint64_t at = words.size() * sizeof(uint32_t);
        words.insert(words.end(), values.begin(), values.end());
        return at;
    };
    m_firstOffset = append(m_regions.Starts());
    m_countsOffset = append(m_regions.Counts());
    m_primsOffset = append(m_regions.Items());
    m_originsOffset = append(origins);
    std::vector<uint32_t> bases(m_worlds + 1, kNoRow);
    std::vector<uint32_t> offsets;
    for (const auto& [world, desc] : m_composed) {
        bases[world] = uint32_t(offsets.size() / 3);
        const float hidden = std::numeric_limits<float>::quiet_NaN();
        std::vector<Float3> shifts(m_worlds, Float3{hidden, hidden, hidden});
        for (size_t at = 0; at < desc.worlds.size(); ++at) {
            if (desc.worlds[at] < m_worlds) {
                const Double3 offset = at < desc.offsets.size() ? desc.offsets[at] : Double3{};
                shifts[desc.worlds[at]] = {float(offset.x), float(offset.y), float(offset.z)};
            }
        }
        const auto* words32 = reinterpret_cast<const uint32_t*>(shifts.data());
        offsets.insert(offsets.end(), words32, words32 + shifts.size() * 3);
    }
    m_offsetBasesOffset = append(bases);
    m_offsetsOffset = append(offsets);
    m_layout =
        device.CreateBuffer({.bytes = words.size() * sizeof(uint32_t), .label = "instance layout"});
    device.GetQueue().Write(*m_layout, words.data());
    m_renderOrigins =
        device.CreateBuffer({.label = "render origins"},
                            std::span<const RenderOrigin>(std::vector<RenderOrigin>(m_worlds + 1)));
    m_viewers = device.CreateBuffer(
        {.label = "viewers"},
        std::span<const uint32_t>(std::vector<uint32_t>(m_worlds + 1, kNoPrim)));
    WriteLayoutHeader();

    m_refits.assign(m_worlds + 1, 0);
    ++m_relayouts;
    Log(LogLevel::Debug,
        "laid out {} TLASes with room for {} instances, {} global prims in each and {} "
        "instancers",
        m_worlds + 1, m_regions.Items().size(), m_globals.size(), m_instancers.size());
}

void Instances::BuildInstancing() {
    const Prims& prims = m_catalog.GetPrims();
    Store& store = m_catalog.GetStore();

    struct Placed {
        uint32_t first = 0;
        uint32_t count = 0;
        uint32_t width = 0;
        uint32_t lights = 0;
    };

    std::vector<PrototypeEntry> entries;
    std::vector<PrototypePiece> pieces;
    std::vector<InstancerBlock> blocks;
    std::vector<uint32_t> ranges;
    std::vector<int64_t> hidden;
    std::map<uint32_t, Placed> placed;
    std::set<uint32_t> building;
    std::map<uint32_t, const Instancer*> byPrim;
    for (const Instancer& instancer : m_instancers) {
        byPrim[instancer.prim] = &instancer;
    }
    m_blockItems.assign(m_worlds + 1, {});
    m_lightCopies.assign(m_worlds + 1, {});
    m_instancerPrims.clear();
    m_watch.clear();

    std::function<std::optional<InstancerBlock>(const Instancer&, uint32_t)> describe;
    const auto place = [&](uint32_t root, uint32_t depth) {
        if (const auto found = placed.find(root); found != placed.end()) {
            return found->second;
        }
        if (building.contains(root)) {
            return Placed{};
        }
        building.insert(root);
        std::vector<PrototypePiece> local;
        uint32_t start = 0;
        uint32_t lightStart = 0;
        for (const uint32_t drawable : m_drawables[root]) {
            const uint32_t light = IsEmitter(uint32_t(prims.At(drawable).schema)) ? 1 : 0;
            local.push_back({.prim = drawable,
                             .start = start,
                             .slots = 1,
                             .nested = kNoNested,
                             .lightStart = lightStart,
                             .lights = light});
            ++start;
            lightStart += light;
        }
        for (const uint32_t nested : m_nested[root]) {
            const auto instancer = byPrim.find(nested);
            if (instancer == byPrim.end()) {
                continue;
            }
            const std::optional<InstancerBlock> inner = describe(*instancer->second, depth + 1);
            if (!inner || inner->count == 0 || inner->width == 0) {
                continue;
            }
            local.push_back({.prim = nested,
                             .start = start,
                             .slots = inner->count * inner->width,
                             .nested = uint32_t(blocks.size()),
                             .lightStart = lightStart,
                             .lights = inner->count * inner->lightWidth});
            blocks.push_back(*inner);
            start += local.back().slots;
            lightStart += local.back().lights;
        }
        building.erase(root);
        const Placed result{.first = uint32_t(pieces.size()),
                            .count = uint32_t(local.size()),
                            .width = start,
                            .lights = lightStart};
        pieces.insert(pieces.end(), local.begin(), local.end());
        placed[root] = result;
        return result;
    };
    describe = [&](const Instancer& instancer, uint32_t depth) -> std::optional<InstancerBlock> {
        if (depth >= kMostNesting) {
            return std::nullopt;
        }
        std::vector<PrototypeEntry> mine;
        uint32_t width = 0;
        uint32_t lightWidth = 0;
        for (const uint32_t root : instancer.roots) {
            PrototypeEntry entry{
                .first = 0, .count = 0, .parent = kNoPrim, .width = 0, .lights = 0};
            if (root != kNoPrim) {
                const Placed pieced = place(root, depth);
                entry.first = pieced.first;
                entry.count = pieced.count;
                entry.width = pieced.width;
                entry.lights = pieced.lights;
                entry.parent = instancer.native ? root : prims.At(root).parent;
                while (entry.parent != kNoPrim &&
                       !IsA(prims.At(entry.parent).schema, SchemaId::Xformable)) {
                    entry.parent = prims.At(entry.parent).parent;
                }
            }
            width = std::max(width, entry.width);
            lightWidth = std::max(lightWidth, entry.lights);
            mine.push_back(entry);
        }
        const uint32_t first = uint32_t(entries.size());
        entries.insert(entries.end(), mine.begin(), mine.end());
        const uint32_t range = uint32_t(ranges.size() / 2);
        ranges.push_back(uint32_t(hidden.size()));
        if (!instancer.native) {
            const ArrayRef ids = store.Array(kPointInstancerInvisibleIds,
                                             prims.Row(instancer.prim, SchemaId::PointInstancer));
            if (ids.address != 0) {
                std::vector<int64_t> sorted =
                    Items<int64_t>(store.Arrays(ColumnType::Int64Array).Read(ids));
                std::ranges::sort(sorted);
                hidden.insert(hidden.end(), sorted.begin(), sorted.end());
            }
        }
        ranges.push_back(uint32_t(hidden.size()) - ranges.back());
        return InstancerBlock{.instancer = instancer.prim,
                              .first = 0,
                              .count = instancer.count,
                              .width = width,
                              .prototypes = first,
                              .prototypeCount = uint32_t(instancer.roots.size()),
                              .native = instancer.native ? 1u : 0u,
                              .hidden = range,
                              .lightWidth = lightWidth};
    };

    for (const Instancer& instancer : m_instancers) {
        m_instancerPrims.insert(instancer.prim);
        if (m_members.contains(instancer.prim)) {
            continue;
        }
        const std::optional<InstancerBlock> block = describe(instancer, 0);
        if (!block || block->width == 0 || block->count == 0) {
            continue;
        }
        std::set<uint32_t> worlds;
        for (const uint32_t root : instancer.roots) {
            if (root != kNoPrim) {
                for (const uint32_t drawable : m_drawables[root]) {
                    worlds.insert(prims.At(drawable).world);
                }
            }
        }
        const auto put = [&](uint32_t tlas) {
            std::vector<uint32_t>& items = m_blockItems[tlas];
            InstancerBlock placedBlock = *block;
            placedBlock.first = uint32_t(items.size());
            const uint32_t index = uint32_t(blocks.size());
            blocks.push_back(placedBlock);
            items.insert(items.end(), size_t(block->count) * block->width, kInstancerSlot | index);
            for (uint32_t offset = 0; offset < block->count * block->lightWidth; ++offset) {
                m_lightCopies[tlas].push_back({index, offset});
            }
            for (const uint32_t world : worlds) {
                if (world != instancer.world) {
                    m_watch[world].insert(tlas);
                }
            }
        };
        if (instancer.world == kGlobalWorld) {
            for (uint32_t tlas = 0; tlas <= m_worlds; ++tlas) {
                if (ShowsGlobals(tlas)) {
                    put(tlas);
                }
            }
        } else if (instancer.world < m_worlds) {
            put(instancer.world);
            if (const auto shown = m_shownIn.find(instancer.world); shown != m_shownIn.end()) {
                for (const uint32_t tlas : shown->second) {
                    put(tlas);
                }
            }
        }
    }
    ++m_version;

    std::vector<std::byte> tables;
    const auto table = [&](auto values) {
        const uint64_t at = RoundUp(tables.size(), 16);
        tables.resize(at);
        const std::span<const std::byte> bytes = std::as_bytes(std::span(values));
        tables.insert(tables.end(), bytes.begin(), bytes.end());
        return at;
    };
    m_blocksOffset = table(blocks);
    m_entriesOffset = table(entries);
    m_piecesOffset = table(pieces);
    m_rangesOffset = table(ranges);
    m_hiddenOffset = table(hidden);
    tables.resize(std::max<size_t>(RoundUp(tables.size(), 16), 16));
    m_instancing =
        store.GetDevice().CreateBuffer({.label = "instancers"}, std::span<const std::byte>(tables));
}

void Instances::WriteLayoutHeader() {
    InstanceLayout layout{};
    layout.handles.address = m_tlas->Handles().Address();
    layout.shapes.address = m_shapes->Address();
    layout.hits.address = m_hits->Address();
    layout.blocks.address = m_instancing->Address() + m_blocksOffset;
    layout.prototypes.address = m_instancing->Address() + m_entriesOffset;
    layout.pieces.address = m_instancing->Address() + m_piecesOffset;
    layout.hiddenRanges.address = m_instancing->Address() + m_rangesOffset;
    layout.hiddenIds = m_instancing->Address() + m_hiddenOffset;
    layout.records = m_records->Address();
    layout.worldCount = m_worlds;
    layout.displayColorName = m_catalog.GetTokens().Intern("displayColor").id;
    layout.firstInstances.address = m_layout->Address() + m_firstOffset;
    layout.instanceCounts.address = m_layout->Address() + m_countsOffset;
    layout.instancePrims.address = m_layout->Address() + m_primsOffset;
    layout.worldOrigins.address = m_layout->Address() + m_originsOffset;
    layout.offsetBases.address = m_layout->Address() + m_offsetBasesOffset;
    layout.offsets.address = m_layout->Address() + m_offsetsOffset;
    m_catalog.GetStore().GetDevice().GetQueue().Write(m_layout->Slice(0, sizeof(layout)), &layout);
}

bool Instances::Restack() {
    BuildInstancing();
    device::Queue& queue = m_catalog.GetStore().GetDevice().GetQueue();
    for (uint32_t tlas = 0; tlas < m_regions.Size(); ++tlas) {
        if (!m_regions.Replace(tlas, RegionItems(tlas))) {
            return false;
        }
        const uint32_t start = m_regions.Starts()[tlas];
        queue.Write(m_layout->Slice(m_primsOffset + start * sizeof(uint32_t),
                                    m_regions.Capacity(tlas) * sizeof(uint32_t)),
                    m_regions.Items().data() + start);
    }
    queue.Write(m_layout->Slice(m_countsOffset, m_regions.Size() * sizeof(uint32_t)),
                m_regions.Counts().data());
    WriteLayoutHeader();
    return true;
}

std::optional<std::set<uint32_t>> Instances::Apply(const ChangeSet& changes) {
    ERNIE_ZONE("Instances/Apply");
    const Prims& prims = m_catalog.GetPrims();
    if (prims.WorldCount() != m_worlds) {
        return std::nullopt;
    }
    std::set<uint32_t> touched;
    const auto touch = [&](uint32_t world) {
        if (world == kGlobalWorld) {
            for (uint32_t tlas = 0; tlas <= m_worlds; ++tlas) {
                touched.insert(tlas);
            }
        } else {
            touched.insert(world);
            if (const auto shown = m_shownIn.find(world); shown != m_shownIn.end()) {
                touched.insert(shown->second.begin(), shown->second.end());
            }
        }
    };
    for (const StructureEntry& entry : changes.apis) {
        if (entry.api == SchemaId::PrototypeAPI) {
            return std::nullopt;
        }
    }
    for (const StructureEntry& entry : changes.removed) {
        if (m_origins.contains(entry.prim) || m_instancerPrims.contains(entry.prim) ||
            m_members.contains(entry.prim) ||
            std::ranges::find(m_products, entry.prim) != m_products.end()) {
            return std::nullopt;
        }
        std::vector<uint32_t>& list = entry.world == kGlobalWorld ? m_globals : m_own[entry.world];
        if (std::erase(list, entry.prim) > 0) {
            touch(entry.world);
        }
    }
    for (const StructureEntry& entry : changes.added) {
        if (!prims.Alive(entry.prim) || prims.Generation(entry.prim) != entry.generation) {
            continue;
        }
        const SchemaId schema = prims.At(entry.prim).schema;
        if (IsA(schema, SchemaId::World) || schema == SchemaId::RenderProduct ||
            schema == SchemaId::PointInstancer ||
            prims.HasApi(entry.prim, SchemaId::PrototypeAPI) || Member(entry.prim)) {
            return std::nullopt;
        }
        if (Drawable(schema)) {
            (entry.world == kGlobalWorld ? m_globals : m_own[entry.world]).push_back(entry.prim);
            touch(entry.world);
        }
    }
    for (const uint32_t tlas : touched) {
        if (!m_regions.Replace(tlas, RegionItems(tlas))) {
            return std::nullopt;
        }
    }

    device::Queue& queue = m_catalog.GetStore().GetDevice().GetQueue();
    for (const uint32_t tlas : touched) {
        const uint32_t start = m_regions.Starts()[tlas];
        queue.Write(m_layout->Slice(m_primsOffset + start * sizeof(uint32_t),
                                    m_regions.Capacity(tlas) * sizeof(uint32_t)),
                    m_regions.Items().data() + start);
    }
    if (!touched.empty()) {
        queue.Write(m_layout->Slice(m_countsOffset, m_regions.Size() * sizeof(uint32_t)),
                    m_regions.Counts().data());
    }
    return touched;
}

device::Operation Instances::Run(const ChangeSet& changes, Stream) {
    ERNIE_ZONE("Instances/Run");
    ++m_runs;
    Store& store = m_catalog.GetStore();
    const uint64_t seen = store.WriteSequence();
    bool relaid = changes.resync || !m_tlas || m_catalog.CompositionVersion() != m_compositions;
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
    const uint32_t tlases = m_regions.Size();
    const uint32_t worlds = m_worlds;

    std::vector<uint32_t> dirty;
    const bool global = relaid || store.LastWrite(kRenderProductCamera) > m_seen ||
                        std::ranges::find(changes.worlds, kGlobalWorld) != changes.worlds.end();
    if (global) {
        dirty.resize(tlases);
        std::iota(dirty.begin(), dirty.end(), 0u);
    } else {
        for (const uint32_t world : changes.worlds) {
            if (world < worlds) {
                dirty.push_back(world);
            }
        }
        for (const uint32_t tlas : structural) {
            if (std::ranges::find(dirty, tlas) == dirty.end()) {
                dirty.push_back(tlas);
            }
        }
        for (const uint32_t world : changes.worlds) {
            if (const auto shown = m_shownIn.find(world); shown != m_shownIn.end()) {
                for (const uint32_t tlas : shown->second) {
                    if (std::ranges::find(dirty, tlas) == dirty.end()) {
                        dirty.push_back(tlas);
                    }
                }
            }
            if (const auto watched = m_watch.find(world); watched != m_watch.end()) {
                for (const uint32_t tlas : watched->second) {
                    if (std::ranges::find(dirty, tlas) == dirty.end()) {
                        dirty.push_back(tlas);
                    }
                }
            }
        }
        if (restacked) {
            for (uint32_t tlas = 0; tlas < tlases; ++tlas) {
                structural.insert(tlas);
                if (std::ranges::find(dirty, tlas) == dirty.end()) {
                    dirty.push_back(tlas);
                }
            }
        }
    }
    const auto written = [&](std::span<const ColumnRef> columns) {
        return std::ranges::any_of(
            columns, [&](ColumnRef column) { return store.LastWrite(column) > m_seen; });
    };
    const bool activated = changes.Wrote(TableId::Imageables) || changes.Wrote(TableId::Meshes) ||
                           written(kActivation);
    const bool placed = relaid || restacked || activated || !structural.empty() ||
                        changes.Wrote(TableId::Transforms) || changes.Wrote(TableId::MeshData) ||
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
        return {};
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
    m_stats.refit = uint32_t(refit.size());

    device::Device& device = store.GetDevice();
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Instances/Write");
    const Ref<device::Buffer> dirtyBuffer = device.CreateBuffer({.label = "dirty TLASes"}, dirty);
    InstanceParams params{};
    params.scene.address = store.DeviceData();
    params.layout.address = m_layout->Address();
    params.instances.address = m_records->Address();
    params.shapes.address = m_shapes->Address();
    params.tlases.address = dirtyBuffer->Address();
    params.hits.address = m_hits->Address();
    params.origins.address = m_renderOrigins->Address();
    params.transforms = changes.tables->transforms;
    params.imageables = changes.tables->imageables;
    params.meshes = changes.tables->meshes;
    params.surfaces.address = m_surfaces->Address();
    params.unitBox = m_unitBox->Handle();

    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    OriginParams origins{};
    origins.scene.address = store.DeviceData();
    origins.layout.address = m_layout->Address();
    origins.products.address = m_productBuffer->Address();
    origins.origins.address = m_renderOrigins->Address();
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
        m_layout->Slice(m_countsOffset, m_regions.Size() * sizeof(uint32_t));
    if (!built.empty()) {
        m_builder.Build(encoder, *m_tlas, *m_records, counts, built);
    }
    if (!refit.empty()) {
        m_builder.Refit(encoder, *m_tlas, *m_records, counts, refit);
    }
    return queue.Submit(encoder.Finish());
}

void Instances::Describe(RenderTables& tables) const {
    tables.instances.address = m_layout ? m_layout->Address() : 0;
    tables.origins.address = m_renderOrigins ? m_renderOrigins->Address() : 0;
    tables.surfaces.address = m_surfaces ? m_surfaces->Address() : 0;
}

void Instances::Hold(std::vector<Ref<const RefCounted>>& held) const {
    for (const Ref<device::Buffer>& buffer :
         {m_layout, m_records, m_shapes, m_hits, m_instancing, m_renderOrigins, m_surfaces}) {
        if (buffer) {
            held.emplace_back(buffer);
        }
    }
    if (m_tlas) {
        held.emplace_back(m_tlas);
    }
    if (m_unitBox) {
        held.emplace_back(m_unitBox);
    }
}

void Instances::Prepare() {
    Kernels();
}

device::Program& Instances::Kernels() {
    if (!m_program) {
        m_program = m_catalog.GetStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.kernels.instances"},
             .library = &kSceneKernels,
             .entries = {"WriteInstances", "GatherViewers", "PlaceOrigins"},
             .label = "instances"});
    }
    return *m_program;
}

}

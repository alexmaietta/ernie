// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/instances.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>

#include "ernie/device/commands.h"
#include "ernie/device/kernels/shared.h"
#include "ernie/device/queue.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/kernels/instances.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {

namespace {

bool Drawable(SchemaId schema) {
    return IsA(schema, SchemaId::Gprim) || IsEmitter(uint32_t(schema));
}

}

bool InstanceTable::ShowsGlobals(uint32_t tlas) const {
    const auto composed = m_composed.find(tlas);
    return composed == m_composed.end() || composed->second.global;
}

std::vector<uint32_t> InstanceTable::RegionItems(uint32_t tlas) const {
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

void InstanceTable::Compose() {
    m_compositions = m_structure.CompositionVersion();
    m_composed.clear();
    m_shownIn.clear();
    for (const auto& [world, desc] : m_structure.Compositions()) {
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

std::vector<InstanceTable::Instancer> InstanceTable::GatherInstancers() const {
    const Prims& prims = m_structure.GetPrims();
    ColumnStore& store = m_structure.GetColumnStore();
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
        const PrimRecord& record = prims.Record(prim);
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

bool InstanceTable::Member(uint32_t prim) const {
    const Prims& prims = m_structure.GetPrims();
    for (uint32_t at = prim; at != kNoPrim; at = prims.Record(at).parent) {
        if (m_roots.contains(at)) {
            return true;
        }
    }
    return false;
}

void InstanceTable::Relayout() {
    ERNIE_ZONE("Instances/Relayout");
    const Prims& prims = m_structure.GetPrims();
    device::Device& device = m_structure.GetColumnStore().GetDevice();
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
        const PrimRecord& record = prims.Record(prim);
        uint32_t root = kNoPrim;
        for (uint32_t at = prim; at != kNoPrim && root == kNoPrim; at = prims.Record(at).parent) {
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
    const uint32_t slots = uint32_t(m_regions.Items().size());
    m_records.Allocate(device, slots);
    m_shapes.Allocate(device, slots);
    m_hits.Allocate(device, slots);
    m_slotWorlds.Allocate(device, slots);
    m_renderOrigins.Allocate(device, m_worlds + 1);
    device::CommandEncoder clear = device.GetQueue().CreateEncoder();
    for (const Ref<device::Buffer>& buffer :
         {m_records.Buffer(), m_shapes.Buffer(), m_hits.Buffer(), m_slotWorlds.Buffer(),
          m_renderOrigins.Buffer()}) {
        clear.Fill(*buffer);
    }
    device.GetQueue().Submit(clear.Finish());

    std::vector<uint32_t> words(sizeof(InstanceLayout) / sizeof(uint32_t));
    const auto append = [&](std::span<const uint32_t> values) {
        const uint64_t at = words.size() * sizeof(uint32_t);
        words.insert(words.end(), values.begin(), values.end());
        return at;
    };
    const uint64_t firstInstancesStart = append(m_regions.Starts());
    const uint64_t countsStart = append(m_regions.Counts());
    const uint64_t primsStart = append(m_regions.Items());
    const uint64_t originsStart = append(origins);
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
    const uint64_t basesStart = append(bases);
    const uint64_t offsetsStart = append(offsets);
    m_layout =
        device.CreateBuffer({.bytes = words.size() * sizeof(uint32_t), .label = "instance layout"});
    device.GetQueue().Write(*m_layout, words.data());
    const uint64_t layout = m_layout->Address();
    m_header.firstInstances.address = layout + firstInstancesStart;
    m_header.instanceCounts.address = layout + countsStart;
    m_header.instancePrims.address = layout + primsStart;
    m_header.worldOrigins.address = layout + originsStart;
    m_header.offsetBases.address = layout + basesStart;
    m_header.offsets.address = layout + offsetsStart;
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

void InstanceTable::BuildInstancing() {
    const Prims& prims = m_structure.GetPrims();
    ColumnStore& store = m_structure.GetColumnStore();

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
            const uint32_t light = IsEmitter(uint32_t(prims.Record(drawable).schema)) ? 1 : 0;
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
                entry.parent = instancer.native ? root : prims.Record(root).parent;
                while (entry.parent != kNoPrim &&
                       !IsA(prims.Record(entry.parent).schema, SchemaId::Xformable)) {
                    entry.parent = prims.Record(entry.parent).parent;
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
                    worlds.insert(prims.Record(drawable).world);
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

    std::vector<std::byte> tables;
    const auto table = [&](auto values) {
        const uint64_t at = RoundUp(tables.size(), 16);
        tables.resize(at);
        const std::span<const std::byte> bytes = std::as_bytes(std::span(values));
        tables.insert(tables.end(), bytes.begin(), bytes.end());
        return at;
    };
    const uint64_t blocksStart = table(blocks);
    const uint64_t entriesStart = table(entries);
    const uint64_t piecesStart = table(pieces);
    const uint64_t rangesStart = table(ranges);
    const uint64_t hiddenStart = table(hidden);
    tables.resize(std::max<size_t>(RoundUp(tables.size(), 16), 16));
    m_instancing =
        store.GetDevice().CreateBuffer({.label = "instancers"}, std::span<const std::byte>(tables));
    const uint64_t instancing = m_instancing->Address();
    m_header.blocks.address = instancing + blocksStart;
    m_header.prototypes.address = instancing + entriesStart;
    m_header.pieces.address = instancing + piecesStart;
    m_header.hiddenRanges.address = instancing + rangesStart;
    m_header.hiddenIds = instancing + hiddenStart;
}

void InstanceTable::WriteLayoutHeader() {
    m_header.handles.address = m_tlas->Handles().Address();
    m_header.shapes.address = m_shapes.Address();
    m_header.hits.address = m_hits.Address();
    m_header.records = m_records.Address();
    m_header.worldCount = m_worlds;
    m_header.displayColorName = m_structure.GetTokens().Intern("displayColor").id;
    m_header.displayOpacityName = m_structure.GetTokens().Intern("displayOpacity").id;
    m_structure.GetColumnStore().GetDevice().GetQueue().Write(m_layout->Slice(0, sizeof(m_header)),
                                                              &m_header);
}

device::BufferRange InstanceTable::LayoutWords(uint64_t address, uint32_t first,
                                               uint32_t count) const {
    return m_layout->Slice(address - m_layout->Address() + uint64_t(first) * sizeof(uint32_t),
                           uint64_t(count) * sizeof(uint32_t));
}

void InstanceTable::WriteRegions(const std::set<uint32_t>& tlases) {
    if (tlases.empty()) {
        return;
    }
    device::Queue& queue = m_structure.GetColumnStore().GetDevice().GetQueue();
    for (const uint32_t tlas : tlases) {
        const uint32_t start = m_regions.Starts()[tlas];
        queue.Write(LayoutWords(m_header.instancePrims.address, start, m_regions.Capacity(tlas)),
                    m_regions.Items().data() + start);
    }
    queue.Write(LayoutWords(m_header.instanceCounts.address, 0, m_regions.Size()),
                m_regions.Counts().data());
}

bool InstanceTable::Restack() {
    BuildInstancing();
    std::set<uint32_t> tlases;
    for (uint32_t tlas = 0; tlas < m_regions.Size(); ++tlas) {
        if (!m_regions.Replace(tlas, RegionItems(tlas))) {
            return false;
        }
        tlases.insert(tlas);
    }
    WriteRegions(tlases);
    WriteLayoutHeader();
    return true;
}

std::optional<std::set<uint32_t>> InstanceTable::Apply(const ChangeSet& changes) {
    ERNIE_ZONE("Instances/Apply");
    const Prims& prims = m_structure.GetPrims();
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
        const SchemaId schema = prims.Record(entry.prim).schema;
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

    WriteRegions(touched);
    return touched;
}

}

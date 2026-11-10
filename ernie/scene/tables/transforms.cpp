// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/transforms.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/captures.h"
#include "ernie/scene/tables/kernels/transforms.h"
#include "ernie/stage/publish/change_set.h"
#include "ernie/stage/views/attribute.h"
#include "ernie/stage/views/view.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

constexpr ColumnRef kFollowInputs[] = {kFollowAPIFollowTarget,
                                       kFollowAPIFollowLookAt,
                                       kFollowAPIFollowTranslate,
                                       kFollowAPIFollowRotate,
                                       kFollowAPIFollowUp,
                                       kFollowAPIFollowInheritScale,
                                       kFollowAPIFollowOffsetTranslate,
                                       kFollowAPIFollowOffsetOrient,
                                       kFollowAPIFollowOffsetScale,
                                       kFollowAPIFollowLookAtOffsetTranslate,
                                       kFollowAPIFollowPositionLag,
                                       kFollowAPIFollowRotationLag,
                                       kFollowAPIFollowLagMaxDistance,
                                       kFollowAPIFollowWeight};

constexpr ColumnRef kInputs[] = {kXformableTranslate,
                                 kXformableOrient,
                                 kXformableScale,
                                 kXformableResetXformStack,
                                 kImageableVisibility,
                                 kImageablePurpose,
                                 kMaterialBindingAPIMaterialBinding,
                                 kMotionAPIMotionVelocity,
                                 kMotionAPIMotionAngularVelocity,
                                 kFollowAPIFollowTarget,
                                 kFollowAPIFollowLookAt,
                                 kFollowAPIFollowTranslate,
                                 kFollowAPIFollowRotate,
                                 kFollowAPIFollowUp,
                                 kFollowAPIFollowInheritScale,
                                 kFollowAPIFollowOffsetTranslate,
                                 kFollowAPIFollowOffsetOrient,
                                 kFollowAPIFollowOffsetScale,
                                 kFollowAPIFollowLookAtOffsetTranslate,
                                 kFollowAPIFollowPositionLag,
                                 kFollowAPIFollowRotationLag,
                                 kFollowAPIFollowLagMaxDistance,
                                 kFollowAPIFollowWeight};

constexpr ColumnRef kStamped[] = {kXformableTranslate,
                                  kXformableOrient,
                                  kXformableScale,
                                  kXformableResetXformStack,
                                  kImageableVisibility,
                                  kImageablePurpose,
                                  kMaterialBindingAPIMaterialBinding};

constexpr TableId kTableOutputs[] = {TableId::Transforms, TableId::Imageables};

constexpr uint32_t kUnresolved = kNoPrim - 1;

}

TransformTable::TransformTable(Structure& structure, const SimulationClock* clock)
    : m_structure(structure), m_clock(clock) {
    for (const ColumnRef column : kStamped) {
        m_structure.GetColumnStore().TrackRows(column);
    }
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
    const bool replanned = rebuild || changes.structure || !changes.apis.empty() ||
                           std::ranges::any_of(kFollowInputs, [&](ColumnRef input) {
                               return store.LastWrite(input) > m_followSeen;
                           });
    bool refollowed = false;
    if (replanned) {
        refollowed = !m_follows.frames.empty();
        PlanFollows();
        m_followSeen = store.WriteSequence();
    }
    if (refollowed && !global) {
        segments.resize(m_segments.Size());
        std::iota(segments.begin(), segments.end(), 0u);
        std::erase_if(segments,
                      [&](uint32_t segment) { return m_segments.Counts()[segment] == 0; });
    }
    const auto written = [&](std::span<const ColumnRef> inputs) {
        return rebuild || std::ranges::any_of(inputs, [&](ColumnRef input) {
                   return store.LastWrite(input) > m_seen;
               });
    };
    const std::span<const ColumnRef> stamped = kStamped;
    const bool moved = written(stamped.first(4));
    const bool following = !m_follows.frames.empty() && (moved || replanned || m_follows.lagged);
    if (segments.empty() && !following) {
        return;
    }
    uint32_t widest = 0;
    for (const uint32_t segment : segments) {
        widest = std::max(widest, m_segments.Counts()[segment]);
        m_dispatched += m_segments.Counts()[segment];
    }

    ResolveParams params{};
    params.tree.parent.address = m_parents.Buffer()->Address();
    params.tree.xform.address = m_xforms.Buffer()->Address();
    params.tree.segmentRows.address = m_segmentRows->Address();
    params.tree.segmentStart.address = m_segmentStart->Address();
    params.tree.segmentCount.address = m_segmentCount->Address();
    params.scene.address = store.DeviceData();
    params.transforms.address = m_transforms.Address();
    params.imageables.address = m_imageables.Address();
    Paged<uint64_t>* stamps[] = {&params.translateStamps,      &params.orientStamps,
                                 &params.scaleStamps,          &params.resetXformStackStamps,
                                 &params.visibilityStamps,     &params.purposeStamps,
                                 &params.materialBindingStamps};
    for (size_t at = 0; at < std::size(kStamped); ++at) {
        const ColumnRef column = kStamped[at];
        stamps[at]->pages.address = store.Schema(column.schema).Pages(store.StampColumn(column));
    }
    params.since = changes.sinceStamp;
    params.until = changes.untilStamp;
    params.all = rebuild || refollowed ? 1 : 0;
    params.inherit = changes.removed.empty() && changes.apis.empty() ? 0 : 1;

    const bool inherited =
        written(stamped.subspan(4)) || !changes.removed.empty() || !changes.apis.empty();
    m_seen = store.WriteSequence();
    std::vector<uint32_t> worlds;
    if (global || refollowed) {
        worlds.push_back(kGlobalWorld);
    } else {
        for (const uint32_t segment : segments) {
            worlds.push_back(SlotWorld(segment));
        }
    }
    if ((moved || refollowed) && !segments.empty()) {
        versions.Record(TableId::Transforms, worlds);
    }
    if (inherited && !segments.empty()) {
        versions.Record(TableId::Imageables, worlds);
    }
    if (following) {
        versions.Record(TableId::Transforms, m_follows.worlds);
    }
    device::Queue& queue = store.GetDevice().GetQueue();
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Hierarchy/Resolve");
    device::CommandEncoder encoder = queue.CreateEncoder();
    Ref<device::Buffer> dirty;
    if (!segments.empty()) {
        dirty = store.GetDevice().CreateBuffer({.label = "hierarchy dirty segments"}, segments);
        params.segments.address = dirty->Address();
        const device::ZoneScope scope(encoder, zone);
        encoder.Dispatch(
            Kernels().Entry("ResolveRows"), params,
            {.x = device::GroupCount(widest, kHierarchyGroup), .y = uint32_t(segments.size())});
    }
    if (following) {
        RecordFollows(encoder, params);
    }
    queue.Submit(encoder.Finish());
}

void TransformTable::PlanFollows() {
    ERNIE_ZONE("Hierarchy/PlanFollows");
    const Prims& prims = m_structure.GetPrims();
    m_follows = {};
    std::vector<uint32_t> followers;
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (prims.Alive(prim) && IsA(prims.Record(prim).schema, SchemaId::Xformable) &&
            prims.HasApi(prim, SchemaId::FollowAPI)) {
            followers.push_back(prim);
        }
    }
    if (followers.empty()) {
        return;
    }

    const auto resolve = [&](const std::vector<std::string>& paths) {
        if (paths.empty()) {
            return kNoPrim;
        }
        const uint32_t prim = m_structure.Find(paths.front());
        const bool valid = prim != kNoPrim && prims.Alive(prim) &&
                           IsA(prims.Record(prim).schema, SchemaId::Xformable);
        return valid ? prim : kNoPrim;
    };

    struct Link {
        uint32_t target = kNoPrim;
        uint32_t lookAt = kNoPrim;
    };

    std::unordered_map<uint32_t, Link> links;
    {
        const View view(m_structure, SchemaId::FollowAPI, followers);
        const UntypedAttribute targets = view.Get(kFollowAPIFollowTarget);
        const UntypedAttribute looks = view.Get(kFollowAPIFollowLookAt);
        const std::vector<Float3> lags = view.Get<Float3>(kFollowAPIFollowPositionLag).Get();
        const std::vector<float> turns = view.Get<float>(kFollowAPIFollowRotationLag).Get();
        for (uint32_t index = 0; index < followers.size(); ++index) {
            const uint32_t target = resolve(targets.GetTargets(index));
            if (target == kNoPrim) {
                continue;
            }
            links[followers[index]] = {target, resolve(looks.GetTargets(index))};
            const Float3 lag = lags[index];
            m_follows.lagged =
                m_follows.lagged || lag.x > 0 || lag.y > 0 || lag.z > 0 || turns[index] > 0;
        }
    }

    std::vector<uint32_t> owner;
    std::unordered_map<uint32_t, uint32_t> levels;
    std::unordered_map<uint32_t, uint8_t> marks;
    std::vector<uint32_t> cyclic;
    std::vector<uint32_t> resolving;
    const auto ownerOf = [&](uint32_t prim) {
        std::vector<uint32_t> chain;
        uint32_t found = kNoPrim;
        for (uint32_t at = prim; at != kNoPrim; at = prims.Record(at).parent) {
            if (owner[at] != kUnresolved) {
                found = owner[at];
                break;
            }
            chain.push_back(at);
            if (links.contains(at)) {
                found = at;
                break;
            }
        }
        for (const uint32_t walked : chain) {
            owner[walked] = found;
        }
        return found;
    };
    std::function<uint32_t(uint32_t)> levelOf;
    const std::function<uint32_t(uint32_t)> follow = [&](uint32_t follower) -> uint32_t {
        const uint8_t mark = marks[follower];
        if (mark == 2) {
            return levels[follower];
        }
        if (mark == 1) {
            const auto loop = std::ranges::find(resolving, follower);
            cyclic.insert(cyclic.end(), loop, resolving.end());
            return 0;
        }
        marks[follower] = 1;
        resolving.push_back(follower);
        const Link link = links.at(follower);
        const uint32_t deepest = std::max(
            {levelOf(prims.Record(follower).parent), levelOf(link.target), levelOf(link.lookAt)});
        resolving.pop_back();
        marks[follower] = 2;
        levels[follower] = deepest + 1;
        return deepest + 1;
    };
    levelOf = [&](uint32_t prim) -> uint32_t {
        if (prim == kNoPrim) {
            return 0;
        }
        const uint32_t found = ownerOf(prim);
        return found == kNoPrim ? 0 : follow(found);
    };
    while (true) {
        owner.assign(prims.Size(), kUnresolved);
        levels.clear();
        marks.clear();
        cyclic.clear();
        for (const auto& entry : links) {
            follow(entry.first);
        }
        if (cyclic.empty()) {
            break;
        }
        std::ranges::sort(cyclic);
        cyclic.erase(std::unique(cyclic.begin(), cyclic.end()), cyclic.end());
        for (const uint32_t follower : cyclic) {
            Log(LogLevel::Warning, "{} follows itself through its targets, so it follows nothing",
                m_structure.GetPaths().Path(follower));
            links.erase(follower);
        }
    }
    if (links.empty()) {
        m_follows.lagged = false;
        return;
    }

    uint32_t deepest = 0;
    for (const auto& entry : levels) {
        deepest = std::max(deepest, entry.second);
    }
    std::vector<std::vector<uint32_t>> frames(deepest + 1);
    std::vector<std::vector<uint32_t>> owned(deepest + 1);
    std::vector<std::vector<uint32_t>> owners(deepest + 1);
    std::set<uint32_t> worlds;
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim) || !IsA(prims.Record(prim).schema, SchemaId::Xformable)) {
            continue;
        }
        const uint32_t found = ownerOf(prim);
        if (found == kNoPrim) {
            continue;
        }
        const uint32_t level = levels.at(found);
        const uint32_t row = prims.Row(prim, SchemaId::Imageable);
        if (found == prim) {
            frames[level].push_back(row);
        } else {
            owned[level].push_back(row);
            owners[level].push_back(prims.Row(found, SchemaId::Imageable));
        }
        worlds.insert(prims.Record(prim).world);
    }
    std::vector<uint32_t> rows;
    std::vector<uint32_t> rowOwners;
    for (uint32_t level = 0; level <= deepest; ++level) {
        m_follows.frameStarts.push_back(uint32_t(rows.size()));
        rows.insert(rows.end(), frames[level].begin(), frames[level].end());
        rowOwners.insert(rowOwners.end(), frames[level].size(), kNoRow);
        m_follows.frames.insert(m_follows.frames.end(), frames[level].begin(), frames[level].end());
    }
    m_follows.frameStarts.push_back(uint32_t(rows.size()));
    for (uint32_t level = 0; level <= deepest; ++level) {
        m_follows.ownedStarts.push_back(uint32_t(rows.size()));
        rows.insert(rows.end(), owned[level].begin(), owned[level].end());
        rowOwners.insert(rowOwners.end(), owners[level].begin(), owners[level].end());
        m_follows.owned.insert(m_follows.owned.end(), owned[level].begin(), owned[level].end());
    }
    m_follows.ownedStarts.push_back(uint32_t(rows.size()));
    m_follows.worlds.assign(worlds.begin(), worlds.end());
    device::Device& device = m_structure.GetColumnStore().GetDevice();
    m_followRows = device.CreateBuffer({.label = "follow rows"}, rows);
    m_followOwners = device.CreateBuffer({.label = "follow owners"}, rowOwners);
}

void TransformTable::RecordFollows(device::CommandEncoder& encoder, const ResolveParams& resolved) {
    ColumnStore& store = m_structure.GetColumnStore();
    device::Device& device = store.GetDevice();
    if (m_followStates.Reserve(device,
                               std::max(store.Schema(SchemaId::FollowAPI).Capacity(), 1u))) {
        encoder.Fill(m_followStates.Buffer()->Slice(0, uint64_t(m_followStates.Rows()) *
                                                           sizeof(FollowState)));
    }
    FollowParams params{};
    params.tree = resolved.tree;
    params.scene = resolved.scene;
    params.transforms = resolved.transforms;
    params.rows.address = m_followRows->Address();
    params.owners.address = m_followOwners->Address();
    params.states.address = m_followStates.Address();
    params.zero = resolved.zero;
    if (m_clock != nullptr) {
        if (m_clock->resets != m_clockResets) {
            params.snap = 1;
        } else if (m_clock->steps != m_clockSteps) {
            params.dt = float(std::max(m_clock->end - m_clockEnd, 0.0));
        }
        m_clockResets = m_clock->resets;
        m_clockSteps = m_clock->steps;
        m_clockEnd = m_clock->end;
    }
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Hierarchy/Follow");
    const device::ZoneScope scope(encoder, zone);
    const uint32_t levels = uint32_t(m_follows.frameStarts.size()) - 1;
    for (uint32_t level = 0; level < levels; ++level) {
        const std::pair<const char*, const std::vector<uint32_t>*> passes[] = {
            {"FollowFrames", &m_follows.frameStarts}, {"FollowOwned", &m_follows.ownedStarts}};
        for (const auto& [entry, starts] : passes) {
            params.first = (*starts)[level];
            params.count = (*starts)[level + 1] - params.first;
            if (params.count == 0) {
                continue;
            }
            encoder.Barrier();
            encoder.Dispatch(Kernels().Entry(entry), params,
                             {.x = device::GroupCount(params.count, kHierarchyGroup)});
        }
    }
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
             .entries = {"ResolveRows", "WriteMotions", "FollowFrames", "FollowOwned"},
             .label = "hierarchy"});
    }
    return *m_program;
}

}

// SPDX-License-Identifier: BSD-3-Clause
/// @file transform_graph.cpp
/// @brief Keeping the transform graph in step with the scene's changes, and evaluating and
/// composing it on the device.
#include "ernie/render/transform_graph.h"

#include "ernie/scene/schema/geom.h"
#include "ernie/scene/tokens.h"
#include "ernie/shaders/device/buffer_ref.h"
#include "ernie/shaders/math/affine.h"
#include "ernie/shaders/math/double_float.h"
#include "ernie/shaders/scene/transform.h"
#include "ernie/util/math.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {
namespace {

using shaders::ComposeParameters;
using shaders::LocalParameters;

// The shaders tell op kinds apart by these, so `XformOpKind` keeps its order.
static_assert(TransformGraph::OpKind(XformOpKind::Translate) == shaders::kTranslate);
static_assert(TransformGraph::OpKind(XformOpKind::Scale) == shaders::kScale);
static_assert(TransformGraph::OpKind(XformOpKind::RotateX) == shaders::kRotateX);
static_assert(TransformGraph::OpKind(XformOpKind::RotateXYZ) == shaders::kRotateXYZ);
static_assert(TransformGraph::OpKind(XformOpKind::Orient) == shaders::kOrient);

/// @return A transform's top three rows, each value split exactly into a high and a low float.
TransformGraph::SplitRows Split(const glm::dmat4& m) {
    TransformGraph::SplitRows rows{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            const double value = m[c][r];
            const auto high = static_cast<float>(value);
            rows.high[r][c] = high;
            rows.low[r][c] = static_cast<float>(value - static_cast<double>(high));
        }
    }
    return rows;
}

const TransformGraph::SplitRows kIdentity = Split(glm::dmat4(1.0));

/// @return Whether a bound column's rows of `floats` floats can hold the op's value.
bool TakesFloats(XformOpKind kind, std::int64_t floats) {
    switch (kind) {
    case XformOpKind::Scale:
        return floats == 1 || floats == 3;
    case XformOpKind::RotateX:
    case XformOpKind::RotateY:
    case XformOpKind::RotateZ:
        return floats == 1;
    case XformOpKind::Orient:
        return floats == 4;
    case XformOpKind::Transform:
        return floats == 16;
    default:
        return floats == 3;
    }
}

/// @return The floats a row of a float column holds.
std::int64_t CountFloats(const DeviceColumn& column) {
    return static_cast<std::int64_t>(column.RowBytes() / sizeof(float));
}

glm::uvec3 Groups(std::size_t threads) {
    return {static_cast<std::uint32_t>(DivideRoundUp<std::size_t>(threads, 64)), 1, 1};
}

}  // namespace

TransformGraph::TransformGraph(DeviceContext& context, Arenas& arenas, const Scene& scene,
                               const ShaderCompiler& compiler)
    : context_(context), arenas_(arenas), scene_(scene), uploader_(context, context.Main()),
      local_(context, compiler.Spirv(WorldModules(), "Local"), "local transforms",
             sizeof(LocalParameters)),
      compose_(context, compiler.Spirv(WorldModules(), "Compose"), "world transforms",
               sizeof(ComposeParameters)) {
    const auto column = [&](std::string_view name, DType type, std::vector<std::int64_t> element) {
        return std::make_unique<DeviceColumn>(arenas, name, type, std::move(element), 0);
    };
    parentColumn_ = column("parents", DType::UInt32, {});
    localColumn_ = column("locals", DType::Float32, {2, 3, 4});
    programColumn_ =
        column("programs", DType::UInt32, {kMaxOps, sizeof(DeviceOp) / sizeof(std::uint32_t)});
    lengthColumn_ = column("program lengths", DType::UInt32, {});
    worlds_ = column("worlds", DType::Float32, {2, 3, 4});
    programmed_.device = column("programmed", DType::UInt32, {});
}

TransformGraph::~TransformGraph() = default;

void TransformGraph::Bind(BucketId bucket, std::string_view attribute, const DeviceColumn& column) {
    const std::optional<XformOpKind> kind = ParseXformOpKind(attribute);
    if (!kind) {
        throw std::invalid_argument(std::string(attribute) + " is no transform op");
    }
    const PrimStore& store = scene_.Store();
    const std::optional<Token> token = store.Tokens().Find(attribute);
    if (column.Type() != DType::Float32 || !TakesFloats(*kind, CountFloats(column)) ||
        column.Rows() != store.GetBucket(bucket).Rows() || !token) {
        throw std::invalid_argument(column.Name() + " does not hold the bucket's " +
                                    std::string(attribute) + " rows");
    }
    if (FindBinding(bucket, token) != nullptr) {
        throw std::invalid_argument(std::string(attribute) + " is bound already");
    }
    bindings_.push_back({bucket, *token, &column, column.Address()});
    // Its prims compose on the device now.
    RefreshBucket(bucket);
}

void TransformGraph::SetOrigin(const glm::dvec3& origin) {
    if (origin == origin_) {
        return;
    }
    origin_ = origin;
    // Level 0 lists the roots.
    if (!levels_.empty()) {
        for (const std::uint32_t root : levels_[0].slots) {
            Refresh(root);
        }
    }
}

std::span<const ShaderModule> TransformGraph::WorldModules() {
    static const std::array modules{shaders::kBufferRef, shaders::kAffine, shaders::kDoubleFloat,
                                    shaders::kTransform};
    return modules;
}

glm::dmat4 TransformGraph::ReadWorld(std::span<const float> worlds, std::uint32_t slot) {
    const std::span<const float> rows = worlds.subspan(std::size_t{slot} * 24, 24);
    glm::dmat4 m(1.0);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            const auto at = static_cast<std::size_t>(r * 4 + c);
            m[c][r] = static_cast<double>(rows[at]) + static_cast<double>(rows[12 + at]);
        }
    }
    return m;
}

std::optional<std::uint32_t> TransformGraph::Node(PrimId prim) const {
    if (prim.slot >= nodes_.size() || !nodes_[prim.slot].live ||
        !(nodes_[prim.slot].prim == prim)) {
        return std::nullopt;
    }
    return prim.slot;
}

void TransformGraph::EnsureSlots(std::uint32_t slots) {
    if (slots <= nodes_.size()) {
        return;
    }
    const std::size_t size = std::max<std::size_t>({slots, 2 * nodes_.size(), 64});
    nodes_.resize(size);
    parents_.resize(size, kNone);
    locals_.resize(size, kIdentity);
    for (DeviceColumn* column : {parentColumn_.get(), localColumn_.get(), worlds_.get()}) {
        column->Resize(size, context_.Main());
    }
}

std::uint32_t TransformGraph::Insert(SlotList& list, std::uint32_t slot) {
    const auto position = static_cast<std::uint32_t>(list.slots.size());
    list.slots.push_back(slot);
    list.dirty.push_back(position);
    return position;
}

std::uint32_t TransformGraph::Unlist(SlotList& list, std::uint32_t NodeState::* position,
                                     std::uint32_t slot) {
    const std::uint32_t at = std::exchange(nodes_[slot].*position, kNone);
    const std::uint32_t last = list.slots.back();
    list.slots.pop_back();
    if (at != list.slots.size()) {
        // The last slot moves into the gap.
        list.slots[at] = last;
        list.dirty.push_back(at);
        nodes_[last].*position = at;
    }
    return at;
}

void TransformGraph::Unprogram(std::uint32_t slot) {
    const std::uint32_t at = Unlist(programmed_, &NodeState::programPosition, slot);
    if (at != programs_.size() - 1) {
        programs_[at] = programs_.back();
        lengths_[at] = lengths_.back();
        dirtyPrograms_.Mark(at);
    }
    programs_.pop_back();
    lengths_.pop_back();
}

const TransformGraph::Binding* TransformGraph::FindBinding(BucketId bucket,
                                                           std::optional<Token> attribute) const {
    if (!attribute) {
        return nullptr;
    }
    const auto found = std::ranges::find_if(bindings_, [&](const Binding& binding) {
        return binding.bucket == bucket && binding.attribute == *attribute;
    });
    return found == bindings_.end() ? nullptr : &*found;
}

std::uint32_t TransformGraph::ParentSlot(std::uint32_t slot) const {
    if (nodes_[slot].resets) {
        return kNone;
    }
    const PrimStore& store = scene_.Store();
    const PathTable& paths = store.Paths();
    for (PathId above = paths.Parent(store.PrimPath(nodes_[slot].prim)); above != kNoPath;
         above = above == kRootPath ? kNoPath : paths.Parent(above)) {
        if (const std::optional<PrimId> parent = store.Find(above)) {
            if (Node(*parent)) {
                return parent->slot;
            }
        }
    }
    return kNone;
}

void TransformGraph::SetLevel(std::uint32_t slot, std::uint32_t level) {
    NodeState& node = nodes_[slot];
    if (node.position != kNone) {
        if (node.level == level) {
            return;
        }
        Unlist(levels_[node.level], &NodeState::position, slot);
    }
    while (levels_.size() <= level) {
        levels_.push_back({{},
                           std::make_unique<DeviceColumn>(arenas_, "level", DType::UInt32,
                                                          std::vector<std::int64_t>{}, 0)});
    }
    node.level = level;
    node.position = Insert(levels_[level], slot);
    for (const std::uint32_t child : node.children) {
        SetLevel(child, level + 1);
    }
}

void TransformGraph::Attach(std::uint32_t slot, std::uint32_t parent) {
    nodes_[slot].parent = parent;
    parents_[slot] = parent;
    dirtyParents_.Mark(slot);
    if (parent != kNone) {
        std::vector<std::uint32_t>& children = nodes_[parent].children;
        nodes_[slot].siblingPosition = static_cast<std::uint32_t>(children.size());
        children.push_back(slot);
    }
    SetLevel(slot, parent == kNone ? 0 : nodes_[parent].level + 1);
}

void TransformGraph::Detach(std::uint32_t slot) {
    NodeState& node = nodes_[slot];
    if (node.parent == kNone) {
        return;
    }
    // The last sibling moves into the gap.
    std::vector<std::uint32_t>& siblings = nodes_[node.parent].children;
    const std::uint32_t at = std::exchange(node.siblingPosition, kNone);
    siblings[at] = siblings.back();
    nodes_[siblings[at]].siblingPosition = at;
    siblings.pop_back();
}

void TransformGraph::Reparent(std::uint32_t slot, std::uint32_t parent) {
    Detach(slot);
    Attach(slot, parent);
    if (nodes_[slot].rooted != (parent == kNone)) {
        Refresh(slot);
    }
}

void TransformGraph::AdoptDescendants(std::uint32_t slot) {
    const PrimStore& store = scene_.Store();
    const PathTable& paths = store.Paths();
    std::vector<PathId> stack;
    paths.ForEachChild(store.PrimPath(nodes_[slot].prim),
                       [&](PathId child) { stack.push_back(child); });
    while (!stack.empty()) {
        const PathId path = stack.back();
        stack.pop_back();
        const std::optional<PrimId> prim = store.Find(path);
        if (!prim) {
            paths.ForEachChild(path, [&](PathId child) { stack.push_back(child); });
        } else if (Node(*prim) && nodes_[prim->slot].parent != slot &&
                   ParentSlot(prim->slot) == slot) {
            // Below it was an ancestor further up; its subtree follows it to the new prim.
            Reparent(prim->slot, slot);
        }
    }
}

void TransformGraph::AddNode(PrimId prim) {
    EnsureSlots(prim.slot + 1);
    nodes_[prim.slot] = {.prim = prim, .live = true, .position = kNone};
    ParseOps(prim.slot);
    Attach(prim.slot, ParentSlot(prim.slot));
    AdoptDescendants(prim.slot);
    Refresh(prim.slot);
    // The device holds nothing for the slot yet, whatever the host's copies compare equal to.
    if (nodes_[prim.slot].programPosition == kNone) {
        dirtyLocals_.Mark(prim.slot);
    }
}

void TransformGraph::RemoveNode(std::uint32_t slot) {
    NodeState& node = nodes_[slot];
    node.live = false;
    Detach(slot);
    Unlist(levels_[node.level], &NodeState::position, slot);
    if (node.programPosition != kNone) {
        Unprogram(slot);
    }
    // Children the scene keeps hang from the nearest ancestor left; removed ones go next.
    const std::vector<std::uint32_t> orphans = std::move(node.children);
    node.children.clear();
    node.ops.clear();
    for (const std::uint32_t orphan : orphans) {
        nodes_[orphan].parent = kNone;
        nodes_[orphan].siblingPosition = kNone;
        if (nodes_[orphan].live) {
            Reparent(orphan, ParentSlot(orphan));
        }
    }
}

void TransformGraph::ParseOps(std::uint32_t slot) {
    NodeState& node = nodes_[slot];
    node.ops.clear();
    node.resets = false;
    if (!IsA(scene_.Schema(node.prim), Xformable::Definition())) {
        return;
    }
    node.resets = ResetsXformStack(scene_, node.prim);
    node.ops = XformOpsInOrder(scene_, node.prim);
}

TransformGraph::Built TransformGraph::BuildProgram(std::uint32_t slot) const {
    const NodeState& node = nodes_[slot];
    const PrimStore& store = scene_.Store();
    const BucketId bucket = store.PrimBucket(node.prim);
    const std::uint32_t row = store.PrimRow(node.prim);
    Built built;
    // A root's local moves it by the origin first, in double, so every world stays near it.
    glm::dmat4 folded =
        node.parent == kNone ? glm::translate(glm::dmat4(1.0), -origin_) : glm::dmat4(1.0);
    const auto push = [&](const DeviceOp& op) {
        if (built.length == kMaxOps) {
            throw std::invalid_argument("'" + scene_.PrimPath(node.prim).String() +
                                        "' has more than " + std::to_string(kMaxOps) +
                                        " ops for the device");
        }
        built.program[built.length++] = op;
    };
    // Runs of the host's ops become one matrix each.
    const auto fold = [&] {
        if (folded != glm::dmat4(1.0) && built.onDevice) {
            push({.kind = shaders::kConstant, .constant = Split(folded)});
            folded = glm::dmat4(1.0);
        }
    };
    for (const XformOp& op : node.ops) {
        const Binding* bound = FindBinding(bucket, op.token);
        if (bound == nullptr) {
            folded = folded * XformOpMatrix(scene_, node.prim, op);
            continue;
        }
        built.onDevice = true;
        fold();
        const DeviceColumn& column = *bound->column;
        const auto floats = static_cast<std::uint32_t>(CountFloats(column));
        push({.kind = OpKind(op.kind),
              .flags = (op.invert ? shaders::kInvert : 0u) | floats << shaders::kFloatsShift,
              .source = {bound->address + row * column.RowBytes()}});
    }
    fold();
    built.local = Split(folded);
    return built;
}

void TransformGraph::Refresh(std::uint32_t slot) {
    const Built built = BuildProgram(slot);
    NodeState& node = nodes_[slot];
    node.rooted = node.parent == kNone;
    if (!built.onDevice) {
        if (locals_[slot] != built.local) {
            locals_[slot] = built.local;
            dirtyLocals_.Mark(slot);
        }
        if (node.programPosition != kNone) {
            Unprogram(slot);
            // The device's last local stands until the upload replaces it.
            dirtyLocals_.Mark(slot);
        }
        return;
    }
    if (node.programPosition == kNone) {
        node.programPosition = Insert(programmed_, slot);
        programs_.emplace_back();
        lengths_.push_back(0);
        dirtyPrograms_.Mark(node.programPosition);
    }
    const std::uint32_t at = node.programPosition;
    if (programs_[at] != built.program || lengths_[at] != built.length) {
        programs_[at] = built.program;
        lengths_[at] = built.length;
        dirtyPrograms_.Mark(at);
    }
}

void TransformGraph::RefreshBucket(BucketId bucket) {
    for (const PrimId prim : rows_.Bucket(bucket)) {
        if (Node(prim)) {
            ParseOps(prim.slot);
            Refresh(prim.slot);
        }
    }
}

void TransformGraph::Replay(const ChangeSet& changes) {
    ERNIE_ZONE("Follow the hierarchy's changes");
    PrimRows::Changes followed = rows_.Follow(scene_.Store(), changes);
    // Removals first, as an added prim may take a removed one's slot. Every removed node dies
    // before any is taken out, so none is reattached only to be removed after.
    std::erase_if(followed.removed, [&](PrimId prim) { return !Node(prim); });
    for (const PrimId prim : followed.removed) {
        nodes_[prim.slot].live = false;
    }
    for (const PrimId prim : followed.removed) {
        RemoveNode(prim.slot);
    }
    for (const PrimId prim : followed.added) {
        AddNode(prim);
    }
    // Bound ops read their rows in place: moved rows read from their new places.
    for (const PrimId prim : followed.moved) {
        if (Node(prim)) {
            Refresh(prim.slot);
        }
    }
}

void TransformGraph::RefreshAuthored(const ChangeSet& changes) {
    const PrimStore& store = scene_.Store();
    const std::optional<Token> order = store.Tokens().Find(tokens::kXformOpOrder);
    for (const ChangeSet::Values& values : changes.values) {
        if (!Any(scene_.Classify(values), Dirty::Transform)) {
            continue;
        }
        const bool reordered = values.attribute == order;
        const Bucket& bucket = store.GetBucket(values.bucket);
        for (const RowRange& range : values.rows) {
            for (std::uint32_t row = range.begin; row < std::min(range.end, bucket.Rows()); ++row) {
                const PrimId prim = bucket.prims[row];
                if (!Node(prim)) {
                    continue;
                }
                // Only a new order can reset the stack.
                if (reordered) {
                    ParseOps(prim.slot);
                    const std::uint32_t parent = ParentSlot(prim.slot);
                    if (parent != nodes_[prim.slot].parent) {
                        Detach(prim.slot);
                        Attach(prim.slot, parent);
                    }
                }
                Refresh(prim.slot);
            }
        }
    }
}

void TransformGraph::UploadDirty() {
    const auto uploadMarked = [&](DirtySlots& dirty, const DeviceColumn& column, const auto& host) {
        for (const std::uint32_t slot : dirty.slots) {
            dirty.marked[slot] = false;
        }
        StageRows(uploader_, column, host, dirty.slots);
        dirty.slots.clear();
    };
    uploadMarked(dirtyParents_, *parentColumn_, parents_);
    uploadMarked(dirtyLocals_, *localColumn_, locals_);
    programColumn_->Resize(programs_.size(), context_.Main());
    lengthColumn_->Resize(lengths_.size(), context_.Main());
    // A program's length changes with it: both upload from the same rows.
    StageRows(uploader_, *lengthColumn_, lengths_, dirtyPrograms_.slots);
    uploadMarked(dirtyPrograms_, *programColumn_, programs_);
    const auto uploadList = [&](SlotList& list) {
        list.device->Resize(list.slots.size(), context_.Main());
        StageRows(uploader_, *list.device, list.slots, list.dirty);
        list.dirty.clear();
    };
    uploadList(programmed_);
    for (SlotList& level : levels_) {
        uploadList(level);
    }
}

void TransformGraph::DispatchLocals(CommandList commands) const {
    local_.Dispatch(commands, Groups(programmed_.slots.size()),
                    LocalParameters{ToBufferRef<shaders::uint>(*programmed_.device),
                                    ToBufferRef<DeviceOp>(*programColumn_),
                                    ToBufferRef<shaders::uint>(*lengthColumn_),
                                    ToBufferRef<shaders::float4>(*localColumn_),
                                    static_cast<std::uint32_t>(programmed_.slots.size())});
}

void TransformGraph::DispatchLevel(CommandList commands, const SlotList& level) const {
    compose_.Dispatch(commands, Groups(level.slots.size()),
                      ComposeParameters{ToBufferRef<shaders::uint>(*level.device),
                                        ToBufferRef<shaders::uint>(*parentColumn_),
                                        ToBufferRef<shaders::float4>(*localColumn_),
                                        ToBufferRef<shaders::float4>(*worlds_),
                                        static_cast<std::uint32_t>(level.slots.size())});
}

Ticket TransformGraph::Update(std::span<const Ticket> waits) {
    ERNIE_ZONE("Update transforms");
    Prepare();
    std::vector<Ticket> after(waits.begin(), waits.end());
    after.push_back(uploader_.Flush(waits));
    return context_.Main().Submit(
        [&](CommandList commands) {
            ERNIE_GPU_ZONE(context_, commands, "Transforms");
            // After earlier readers of the worlds and writers of the bound columns.
            ComputeBarrier(commands);
            if (!programmed_.slots.empty()) {
                ERNIE_GPU_ZONE(context_, commands, "Local transforms");
                DispatchLocals(commands);
            }
            for (const SlotList& level : levels_) {
                if (!level.slots.empty()) {
                    ComputeBarrier(commands);
                    ERNIE_GPU_ZONE(context_, commands, "World transforms");
                    DispatchLevel(commands, level);
                }
            }
        },
        after);
}

/// The columns an update uploads come first in `uploaded`, in this order, then each level's slots.
struct TransformGraph::GraphColumns {
    enum Uploaded : std::size_t { kParents, kLocals, kPrograms, kLengths, kProgrammed, kLevels };
    std::vector<graph::Buffer> uploaded;
    graph::Buffer worlds;
    std::vector<graph::Buffer> bound;

    graph::Buffer& Level(std::size_t level) {
        return uploaded[kLevels + level];
    }
};

void TransformGraph::AddUpdate(graph::Graph& graph, bool compose) {
    ERNIE_ZONE("Update transforms");
    Prepare();
    if (nodes_.empty()) {
        return;
    }
    GraphColumns columns = ImportColumns(graph);
    const graph::Graph::ScopeGuard scope = graph.Scope("Transforms");
    graph::AddUploadPass(graph, "Upload transforms", uploader_, columns.uploaded);
    if (compose) {
        AddLocalPass(graph, columns);
        AddLevelPasses(graph, columns);
    }
}

TransformGraph::GraphColumns TransformGraph::ImportColumns(graph::Graph& graph) const {
    // Empty columns hold nothing to read or write: their handles stay invalid.
    const auto import = [&](const DeviceColumn& column) {
        return column.Rows() == 0 ? graph::Buffer{} : graph.Import(column);
    };
    GraphColumns columns;
    for (const DeviceColumn* column :
         {parentColumn_.get(), localColumn_.get(), programColumn_.get(), lengthColumn_.get(),
          programmed_.device.get()}) {
        columns.uploaded.push_back(import(*column));
    }
    for (const SlotList& level : levels_) {
        columns.uploaded.push_back(import(*level.device));
    }
    columns.worlds = import(*worlds_);
    for (const Binding& binding : bindings_) {
        columns.bound.push_back(import(*binding.column));
    }
    return columns;
}

void TransformGraph::AddLocalPass(graph::Graph& graph, GraphColumns& columns) const {
    if (programmed_.slots.empty()) {
        return;
    }
    std::vector<graph::Buffer>& uploaded = columns.uploaded;
    graph.Add(
        "Local transforms", graph::Kind::Compute,
        [&](graph::PassBuilder& pass) {
            for (const std::size_t read :
                 {GraphColumns::kProgrammed, GraphColumns::kPrograms, GraphColumns::kLengths}) {
                pass.ReadAddress(uploaded[read]);
            }
            // Bound columns are read in place, after the passes that write them.
            for (const graph::Buffer& column : columns.bound) {
                pass.ReadAddress(column);
            }
            uploaded[GraphColumns::kLocals] = pass.WriteAddress(uploaded[GraphColumns::kLocals]);
        },
        [this](graph::PassContext& pass) { DispatchLocals(pass.commands); });
}

void TransformGraph::AddLevelPasses(graph::Graph& graph, GraphColumns& columns) const {
    for (std::size_t l = 0; l < levels_.size(); ++l) {
        if (levels_[l].slots.empty()) {
            continue;
        }
        // Each level reads the worlds the level above wrote.
        graph.Add(
            "World transforms", graph::Kind::Compute,
            [&](graph::PassBuilder& pass) {
                for (const graph::Buffer& read :
                     {columns.Level(l), columns.uploaded[GraphColumns::kParents],
                      columns.uploaded[GraphColumns::kLocals]}) {
                    pass.ReadAddress(read);
                }
                columns.worlds = pass.WriteAddress(columns.worlds);
            },
            [this, &level = levels_[l]](graph::PassContext& pass) {
                DispatchLevel(pass.commands, level);
            });
    }
}

void TransformGraph::Prepare() {
    const PrimStore& store = scene_.Store();
    // A bound column that moved to a larger range moves its programs' sources.
    std::vector<BucketId> moved;
    for (Binding& binding : bindings_) {
        if (binding.column->Rows() != store.GetBucket(binding.bucket).Rows()) {
            throw std::logic_error("Sync " + binding.column->Name() +
                                   " with its bucket before updating transforms");
        }
        if (const std::uint64_t address = binding.column->Address(); address != binding.address) {
            binding.address = address;
            moved.push_back(binding.bucket);
        }
    }
    const ChangeSet changes = store.ChangesSince(cursor_);
    cursor_ = changes.version;
    Replay(changes);
    RefreshAuthored(changes);
    for (const BucketId bucket : moved) {
        RefreshBucket(bucket);
    }
    UploadDirty();
}

}  // namespace ernie

// SPDX-License-Identifier: BSD-3-Clause
/// @file transform_graph.cpp
/// @brief Keeping the transform graph in step with the scene's changes, and evaluating and
/// composing it on the device.
#include "ernie/render/transform_graph.h"

#include "ernie/scene/schema/geom.h"
#include "ernie/scene/tokens.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {
namespace {

const ShaderModule kAffine{"affine", R"(
/// A 3x4 affine transform's rows; its fourth row is (0, 0, 0, 1).
public struct Affine {
    public float4 rows[3];
};

public Affine Identity() {
    Affine m;
    m.rows[0] = float4(1, 0, 0, 0);
    m.rows[1] = float4(0, 1, 0, 0);
    m.rows[2] = float4(0, 0, 1, 0);
    return m;
}

public Affine Load(float4* matrices, uint index) {
    Affine m;
    for (uint r = 0; r < 3; ++r) {
        m.rows[r] = matrices[3 * index + r];
    }
    return m;
}

public void Store(float4* matrices, uint index, Affine m) {
    for (uint r = 0; r < 3; ++r) {
        matrices[3 * index + r] = m.rows[r];
    }
}

public Affine Mul(Affine a, Affine b) {
    Affine c;
    for (uint r = 0; r < 3; ++r) {
        float4 row = a.rows[r];
        c.rows[r] = row.x * b.rows[0] + row.y * b.rows[1] + row.z * b.rows[2] +
                    float4(0, 0, 0, row.w);
    }
    return c;
}
)"};

// A program's op kinds: a folded constant, then each `XformOpKind` after it.
constexpr std::uint32_t kConstant = 0;
constexpr std::uint32_t kInvert = 1;
constexpr std::uint32_t kFloatsShift = 8;

constexpr std::uint32_t ProgramKind(XformOpKind kind) {
    return 1 + static_cast<std::uint32_t>(kind);
}

/// The op kinds and the program layout, from the constants above, for the kernels to import.
ShaderModule OpConstants() {
    std::string source;
    const auto constant = [&](const char* name, std::uint32_t value) {
        source +=
            std::string("public static const uint ") + name + " = " + std::to_string(value) + ";\n";
    };
    constant("kConstant", kConstant);
    constant("kTranslate", ProgramKind(XformOpKind::Translate));
    constant("kScale", ProgramKind(XformOpKind::Scale));
    constant("kRotateX", ProgramKind(XformOpKind::RotateX));
    constant("kRotateXYZ", ProgramKind(XformOpKind::RotateXYZ));
    constant("kOrient", ProgramKind(XformOpKind::Orient));
    constant("kInvert", kInvert);
    constant("kFloatsShift", kFloatsShift);
    constant("kMaxOps", TransformGraph::kMaxOps);
    constant("kNone", TransformGraph::kNone);
    return {"ops", std::move(source)};
}

const ShaderModule kLocal{"local", R"(
import affine;
import ops;
struct Op { uint kind; uint flags; float* source; float4 constant[3]; };
struct Parameters { uint* slots; Op* programs; uint* lengths; float4* locals; uint count; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

static const uint3 kAxes[6] = { uint3(0, 1, 2), uint3(0, 2, 1), uint3(1, 0, 2),
                                uint3(1, 2, 0), uint3(2, 0, 1), uint3(2, 1, 0) };

/// A right-handed turn about one axis, as the host's AxisRotation builds it.
Affine Turn(uint axis, float degrees) {
    float c = cos(radians(degrees));
    float s = sin(radians(degrees));
    uint a = (axis + 1) % 3;
    uint b = (axis + 2) % 3;
    Affine m = Identity();
    m.rows[a][a] = c;
    m.rows[a][b] = -s;
    m.rows[b][a] = s;
    m.rows[b][b] = c;
    return m;
}

Affine Inverse(Affine m) {
    float3 a = m.rows[0].xyz;
    float3 b = m.rows[1].xyz;
    float3 c = m.rows[2].xyz;
    float3 r0 = cross(b, c);
    float3 r1 = cross(c, a);
    float3 r2 = cross(a, b);
    float invDet = 1 / dot(a, r0);
    // The inverse's columns are the cofactor rows over the determinant.
    float3x3 inv = transpose(float3x3(r0, r1, r2)) * invDet;
    float3 t = float3(m.rows[0].w, m.rows[1].w, m.rows[2].w);
    float3 back = -mul(inv, t);
    Affine result;
    for (uint r = 0; r < 3; ++r) {
        result.rows[r] = float4(inv[r], back[r]);
    }
    return result;
}

Affine OpMatrix(Op op) {
    Affine m = Identity();
    float* v = op.source;
    uint floats = (op.flags >> kFloatsShift) & 0xff;
    if (op.kind == kConstant) {
        m.rows = op.constant;
    } else if (op.kind == kTranslate) {
        m.rows[0].w = v[0];
        m.rows[1].w = v[1];
        m.rows[2].w = v[2];
    } else if (op.kind == kScale) {
        float3 s = floats == 1 ? float3(v[0]) : float3(v[0], v[1], v[2]);
        m.rows[0].x = s.x;
        m.rows[1].y = s.y;
        m.rows[2].z = s.z;
    } else if (op.kind < kRotateXYZ) {
        m = Turn(op.kind - kRotateX, v[0]);
    } else if (op.kind < kOrient) {
        // The order's first axis turns first: R = Rk * Rj * Ri, each by its own axis' angle.
        uint3 axes = kAxes[op.kind - kRotateXYZ];
        m = Mul(Mul(Turn(axes.z, v[axes.z]), Turn(axes.y, v[axes.y])),
                Turn(axes.x, v[axes.x]));
    } else if (op.kind == kOrient) {
        float4 q = normalize(float4(v[0], v[1], v[2], v[3]));
        m.rows[0] = float4(1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.z * q.w),
                           2 * (q.x * q.z + q.y * q.w), 0);
        m.rows[1] = float4(2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x * q.x + q.z * q.z),
                           2 * (q.y * q.z - q.x * q.w), 0);
        m.rows[2] = float4(2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w),
                           1 - 2 * (q.x * q.x + q.y * q.y), 0);
    } else {
        // Column-major, as the scene's matrices are.
        for (uint r = 0; r < 3; ++r) {
            m.rows[r] = float4(v[r], v[4 + r], v[8 + r], v[12 + r]);
        }
    }
    return (op.flags & kInvert) != 0 ? Inverse(m) : m;
}

/// One thread a programmed node, its program at its place in the list.
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint at = id.x;
    if (at >= parameters.count) {
        return;
    }
    Affine local = Identity();
    for (uint k = 0; k < parameters.lengths[at]; ++k) {
        local = Mul(local, OpMatrix(parameters.programs[at * kMaxOps + k]));
    }
    Store(parameters.locals, parameters.slots[at], local);
}
)"};

const ShaderModule kCompose{"compose", R"(
import affine;
import ops;
struct Parameters { uint* slots; uint* parents; float4* locals; float4* worlds; uint count; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= parameters.count) {
        return;
    }
    uint slot = parameters.slots[id.x];
    uint parent = parameters.parents[slot];
    Affine local = Load(parameters.locals, slot);
    Store(parameters.worlds, slot,
          parent == kNone ? local : Mul(Load(parameters.worlds, parent), local));
}
)"};

struct LocalParameters {
    VkDeviceAddress slots, programs, lengths, locals;
    std::uint32_t count, pad;
};
struct ComposeParameters {
    VkDeviceAddress slots, parents, locals, worlds;
    std::uint32_t count, pad;
};
static_assert(sizeof(TransformGraph::DeviceOp) == 64);

std::vector<std::uint32_t> Compile(const ShaderCompiler& compiler, const ShaderModule& kernel) {
    const std::array modules{kAffine, OpConstants(), kernel};
    return compiler.Spirv(modules, "main");
}

TransformGraph::Rows12 TopRows(const glm::dmat4& m) {
    TransformGraph::Rows12 rows{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            rows[static_cast<std::size_t>(r * 4 + c)] = static_cast<float>(m[c][r]);
        }
    }
    return rows;
}

const TransformGraph::Rows12 kIdentity = TopRows(glm::dmat4(1.0));

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
std::int64_t Floats(const DeviceColumn& column) {
    return static_cast<std::int64_t>(column.RowBytes() / sizeof(float));
}

glm::uvec3 Groups(std::size_t threads) {
    return {static_cast<std::uint32_t>((threads + 63) / 64), 1, 1};
}

bool IsPlaceholder(PrimId prim) {
    return prim.slot == PrimId().slot;
}

}  // namespace

TransformGraph::TransformGraph(DeviceContext& context, Arenas& arenas, const Scene& scene,
                               const ShaderCompiler& compiler)
    : context_(context), arenas_(arenas), scene_(scene), uploader_(context, context.Main()),
      local_(context, Compile(compiler, kLocal), "local transforms", sizeof(LocalParameters)),
      compose_(context, Compile(compiler, kCompose), "world transforms",
               sizeof(ComposeParameters)) {
    const auto column = [&](std::string_view name, DType type, std::vector<std::int64_t> element) {
        return std::make_unique<DeviceColumn>(arenas, name, type, std::move(element), 0);
    };
    parentColumn_ = column("parents", DType::UInt32, {});
    localColumn_ = column("locals", DType::Float32, {3, 4});
    programColumn_ = column("programs", DType::UInt32, {kMaxOps, 16});
    lengthColumn_ = column("program lengths", DType::UInt32, {});
    worlds_ = column("worlds", DType::Float32, {3, 4});
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
    if (column.Type() != DType::Float32 || !TakesFloats(*kind, Floats(column)) ||
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
            Detach(prim->slot);
            Attach(prim->slot, slot);
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
            Attach(orphan, ParentSlot(orphan));
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
    const TokenTable& tokens = scene_.Store().Tokens();
    for (XformOp& op : XformOpsInOrder(scene_, node.prim)) {
        const std::optional<Token> attribute = tokens.Find(op.attribute);
        node.ops.push_back({std::move(op), attribute});
    }
}

TransformGraph::Built TransformGraph::BuildProgram(std::uint32_t slot) const {
    const NodeState& node = nodes_[slot];
    const PrimStore& store = scene_.Store();
    const BucketId bucket = store.PrimBucket(node.prim);
    const std::uint32_t row = store.PrimRow(node.prim);
    Built built;
    glm::dmat4 folded(1.0);
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
            push({.kind = kConstant, .constant = TopRows(folded)});
            folded = glm::dmat4(1.0);
        }
    };
    for (const NodeOp& op : node.ops) {
        const Binding* bound = FindBinding(bucket, op.attribute);
        if (bound == nullptr) {
            folded = folded * XformOpMatrix(scene_, node.prim, op.op);
            continue;
        }
        built.onDevice = true;
        fold();
        const DeviceColumn& column = *bound->column;
        const auto floats = static_cast<std::uint32_t>(Floats(column));
        push({.kind = ProgramKind(op.op.kind),
              .flags = (op.op.invert ? kInvert : 0u) | floats << kFloatsShift,
              .source = bound->address + row * column.RowBytes()});
    }
    fold();
    built.local = TopRows(folded);
    return built;
}

void TransformGraph::Refresh(std::uint32_t slot) {
    const Built built = BuildProgram(slot);
    NodeState& node = nodes_[slot];
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
    if (bucket >= rows_.size()) {
        return;
    }
    for (const PrimId prim : rows_[bucket]) {
        if (Node(prim)) {
            ParseOps(prim.slot);
            Refresh(prim.slot);
        }
    }
}

void TransformGraph::Replay(const ChangeSet& changes) {
    ERNIE_ZONE("Follow the hierarchy's changes");
    const PrimStore& store = scene_.Store();
    rows_.resize(store.BucketCount());
    std::vector<PrimId> removed;
    std::vector<PrimId> moved;
    std::vector<BucketId> touched;
    for (const ChangeSet::Rows& entry : changes.rows) {
        std::vector<PrimId>& rows = rows_[entry.bucket];
        const StructuralChange& change = entry.change;
        touched.push_back(entry.bucket);
        switch (change.kind) {
        case StructuralChange::Kind::Added:
            rows.resize(change.row + change.count, PrimId());
            break;
        case StructuralChange::Kind::Removed:
            if (!IsPlaceholder(rows[change.row])) {
                removed.push_back(rows[change.row]);
            }
            // Unless it was the last, a Moved change brings the last row here next.
            if (change.row + 1 == rows.size()) {
                rows.pop_back();
            }
            break;
        case StructuralChange::Kind::Moved:
            rows[change.row] = rows[change.from];
            rows.pop_back();
            if (!IsPlaceholder(rows[change.row])) {
                moved.push_back(rows[change.row]);
            }
            break;
        case StructuralChange::Kind::Reset:
            for (const PrimId prim : rows) {
                if (!IsPlaceholder(prim)) {
                    removed.push_back(prim);
                }
            }
            rows.assign(change.count, PrimId());
            break;
        }
    }
    // Removals first, as an added prim may take a removed one's slot. Every removed node dies
    // before any is taken out, so none is reattached only to be removed after.
    std::erase_if(removed, [&](PrimId prim) { return !Node(prim); });
    for (const PrimId prim : removed) {
        nodes_[prim.slot].live = false;
    }
    for (const PrimId prim : removed) {
        RemoveNode(prim.slot);
    }
    std::vector<PrimId> added;
    std::ranges::sort(touched);
    touched.erase(std::ranges::unique(touched).begin(), touched.end());
    for (const BucketId bucket : touched) {
        std::vector<PrimId>& rows = rows_[bucket];
        const std::vector<PrimId>& now = store.GetBucket(bucket).prims;
        for (std::size_t row = 0; row < rows.size(); ++row) {
            if (IsPlaceholder(rows[row])) {
                rows[row] = now[row];
                added.push_back(now[row]);
            }
        }
    }
    // Parents before children: ancestors' paths are interned first.
    std::ranges::sort(added, {}, [&](PrimId prim) { return store.PrimPath(prim); });
    for (const PrimId prim : added) {
        AddNode(prim);
    }
    // Bound ops read their rows in place: moved rows read from their new places.
    for (const PrimId prim : moved) {
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
    // Changed rows of a host array, each once; the rows are left for reuse.
    const auto upload = [&](const DeviceColumn& column, const auto& host,
                            std::vector<std::uint32_t>& rows) {
        std::ranges::sort(rows);
        rows.erase(std::ranges::unique(rows).begin(), rows.end());
        std::erase_if(rows, [&](std::uint32_t row) { return row >= host.size(); });
        uploader_.WriteRows(column.Storage(), column.Range().offset, column.RowBytes(),
                            std::as_bytes(std::span(host)), rows);
    };
    const auto uploadMarked = [&](DirtySlots& dirty, const DeviceColumn& column, const auto& host) {
        for (const std::uint32_t slot : dirty.slots) {
            dirty.marked[slot] = false;
        }
        upload(column, host, dirty.slots);
        dirty.slots.clear();
    };
    uploadMarked(dirtyParents_, *parentColumn_, parents_);
    uploadMarked(dirtyLocals_, *localColumn_, locals_);
    programColumn_->Resize(programs_.size(), context_.Main());
    lengthColumn_->Resize(lengths_.size(), context_.Main());
    upload(*lengthColumn_, lengths_, dirtyPrograms_.slots);
    uploadMarked(dirtyPrograms_, *programColumn_, programs_);
    const auto uploadList = [&](SlotList& list) {
        list.device->Resize(list.slots.size(), context_.Main());
        upload(*list.device, list.slots, list.dirty);
        list.dirty.clear();
    };
    uploadList(programmed_);
    for (SlotList& level : levels_) {
        uploadList(level);
    }
}

void TransformGraph::RecordDispatches(VkCommandBuffer commands) const {
    ERNIE_GPU_ZONE(context_, commands, "Transforms");
    // After earlier readers of the worlds and writers of the bound columns.
    ComputeBarrier(commands);
    if (!programmed_.slots.empty()) {
        ERNIE_GPU_ZONE(context_, commands, "Local transforms");
        local_.Dispatch(commands, Groups(programmed_.slots.size()),
                        LocalParameters{programmed_.device->Address(), programColumn_->Address(),
                                        lengthColumn_->Address(), localColumn_->Address(),
                                        static_cast<std::uint32_t>(programmed_.slots.size()), 0});
    }
    for (const SlotList& level : levels_) {
        if (level.slots.empty()) {
            continue;
        }
        ComputeBarrier(commands);
        ERNIE_GPU_ZONE(context_, commands, "World transforms");
        compose_.Dispatch(commands, Groups(level.slots.size()),
                          ComposeParameters{level.device->Address(), parentColumn_->Address(),
                                            localColumn_->Address(), worlds_->Address(),
                                            static_cast<std::uint32_t>(level.slots.size()), 0});
    }
}

Ticket TransformGraph::Update(std::span<const Ticket> waits) {
    ERNIE_ZONE("Update transforms");
    const PrimStore& store = scene_.Store();
    // A bound column that moved to a larger range moves its programs' sources.
    std::vector<BucketId> moved;
    for (Binding& binding : bindings_) {
        if (binding.column->Rows() != store.GetBucket(binding.bucket).Rows()) {
            throw std::logic_error("Sync " + binding.column->Name() +
                                   " with its bucket before updating transforms");
        }
        if (const VkDeviceAddress address = binding.column->Address(); address != binding.address) {
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

    std::vector<Ticket> after(waits.begin(), waits.end());
    after.push_back(uploader_.Flush(waits));
    return context_.Main().Submit([&](VkCommandBuffer commands) { RecordDispatches(commands); },
                                  after);
}

}  // namespace ernie

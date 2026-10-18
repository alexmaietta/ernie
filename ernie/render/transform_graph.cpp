// SPDX-License-Identifier: BSD-3-Clause
/// @file transform_graph.cpp
/// @brief Building the transform graph from the scene's hierarchy, and composing it level by
/// level on the device.
#include "ernie/render/transform_graph.h"

#include "ernie/scene/schemas/geom.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ernie {
namespace {

const ShaderModule kAffine{"affine", R"(
/// A 3x4 affine transform's rows; its fourth row is (0, 0, 0, 1).
public struct Affine {
    public float4 rows[3];
};

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

const ShaderModule kPose{"pose", R"(
import affine;
struct Parameters {
    float* translate; float4* orient; uint* nodes; float4* before; float4* after;
    float4* locals; uint rows;
};
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;
    if (row >= parameters.rows) {
        return;
    }
    uint node = parameters.nodes[row];
    if (node == 0xffffffff) {
        return;
    }
    float3 t = float3(parameters.translate[3 * row], parameters.translate[3 * row + 1],
                      parameters.translate[3 * row + 2]);
    float4 q = normalize(parameters.orient[row]);
    Affine pose;
    pose.rows[0] = float4(1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.z * q.w),
                          2 * (q.x * q.z + q.y * q.w), t.x);
    pose.rows[1] = float4(2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x * q.x + q.z * q.z),
                          2 * (q.y * q.z - q.x * q.w), t.y);
    pose.rows[2] = float4(2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w),
                          1 - 2 * (q.x * q.x + q.y * q.y), t.z);
    Affine local = Mul(Mul(Load(parameters.before, node), pose), Load(parameters.after, node));
    Store(parameters.locals, node, local);
}
)"};

const ShaderModule kCompose{"compose", R"(
import affine;
struct Parameters { uint* parents; float4* locals; float4* worlds; uint first; uint count; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= parameters.count) {
        return;
    }
    uint node = parameters.first + id.x;
    uint parent = parameters.parents[node];
    Affine local = Load(parameters.locals, node);
    Store(parameters.worlds, node,
          parent == 0xffffffff ? local : Mul(Load(parameters.worlds, parent), local));
}
)"};

struct PoseParameters {
    VkDeviceAddress translate, orient, nodes, before, after, locals;
    std::uint32_t rows, pad;
};
struct ComposeParameters {
    VkDeviceAddress parents, locals, worlds;
    std::uint32_t first, count;
};

std::vector<std::uint32_t> Compile(const ShaderCompiler& compiler, const ShaderModule& kernel) {
    const std::array modules{kAffine, kernel};
    return compiler.Spirv(modules, "main");
}

/// A host transform's top three rows, as the device keeps them.
TransformGraph::Rows12 Rows(const glm::dmat4& m) {
    TransformGraph::Rows12 rows{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            rows[static_cast<std::size_t>(r * 4 + c)] = static_cast<float>(m[c][r]);
        }
    }
    return rows;
}

glm::uvec3 Groups(std::uint32_t threads) {
    return {(threads + 63) / 64, 1, 1};
}

}  // namespace

TransformGraph::TransformGraph(DeviceContext& context, Arenas& arenas, const Scene& scene,
                               const ShaderCompiler& compiler)
    : context_(context), arenas_(arenas), scene_(scene), uploader_(context, context.Main()),
      pose_(context, Compile(compiler, kPose), "pose", sizeof(PoseParameters)),
      compose_(context, Compile(compiler, kCompose), "compose", sizeof(ComposeParameters)) {
    const auto column = [&](std::string_view name, DType type, std::vector<std::int64_t> element) {
        return std::make_unique<DeviceColumn>(arenas, name, type, std::move(element), 0);
    };
    parents_ = column("parents", DType::UInt32, {});
    locals_ = column("locals", DType::Float32, {3, 4});
    before_ = column("before poses", DType::Float32, {3, 4});
    after_ = column("after poses", DType::Float32, {3, 4});
    worlds_ = column("worlds", DType::Float32, {3, 4});
    slotNodes_ = column("slot nodes", DType::UInt32, {});
}

TransformGraph::~TransformGraph() = default;

void TransformGraph::BindPoses(BucketId bucket, const DeviceColumn& translate,
                               const DeviceColumn& orient) {
    const std::uint32_t rows = scene_.Store().GetBucket(bucket).Rows();
    const auto fits = [&](const DeviceColumn& column, std::int64_t components) {
        return column.Rows() == rows && column.Type() == DType::Float32 &&
               column.Element() == std::vector<std::int64_t>{components};
    };
    if (!fits(translate, 3) || !fits(orient, 4)) {
        throw std::invalid_argument("Poses take a float3 and a float4 a row of the bucket");
    }
    if (FindPoses(bucket) != nullptr) {
        throw std::invalid_argument("The bucket's poses are bound already");
    }
    poses_.push_back({bucket, &translate, &orient,
                      std::make_unique<DeviceColumn>(arenas_, "pose nodes", DType::UInt32,
                                                     std::vector<std::int64_t>{}, 0)});
    // Its rows' locals come from the device now.
    built_ = false;
}

const TransformGraph::Poses* TransformGraph::FindPoses(BucketId bucket) const {
    const auto found = std::ranges::find(poses_, bucket, &Poses::bucket);
    return found == poses_.end() ? nullptr : &*found;
}

std::optional<std::uint32_t> TransformGraph::Node(PrimId prim) const {
    if (prim.slot >= slotNode_.size() || slotNode_[prim.slot] == kNone ||
        !scene_.Store().Alive(prim)) {
        return std::nullopt;
    }
    return slotNode_[prim.slot];
}

void TransformGraph::Write(DeviceColumn& column, std::vector<Rows12>& uploaded, std::uint32_t node,
                           const Rows12& rows) {
    // Changes come by chunk of rows: only nodes whose transform did change go up.
    if (uploaded[node] == rows) {
        return;
    }
    uploaded[node] = rows;
    uploader_.Write(column.Storage(), column.Range().offset + node * column.RowBytes(),
                    std::span<const float>(rows));
}

void TransformGraph::UploadLocal(PrimId prim, std::uint32_t node, const Poses* poses) {
    const bool xformable = IsA(scene_.Schema(prim), Xformable::Definition());
    const std::optional<PoseOps> around =
        poses != nullptr && xformable ? SplitAroundPose(scene_, prim) : std::nullopt;
    if (around) {
        Write(*before_, uploadedBefore_, node, Rows(around->before));
        Write(*after_, uploadedAfter_, node, Rows(around->after));
    } else {
        Write(*locals_, uploadedLocals_, node, Rows(scene_.LocalTransform(prim)));
    }
}

void TransformGraph::Rebuild() {
    const PrimStore& store = scene_.Store();
    const PathTable& paths = store.Paths();
    std::vector<PrimId> prims;
    std::uint32_t slots = 0;
    for (BucketId bucket = 0; bucket < store.BucketCount(); ++bucket) {
        for (const PrimId prim : store.GetBucket(bucket).prims) {
            prims.push_back(prim);
            slots = std::max(slots, prim.slot + 1);
        }
    }
    // Ancestors' paths are interned before their descendants', so by path each prim comes after
    // its parent and its depth follows from the parent's.
    std::ranges::sort(prims, {}, [&](PrimId prim) { return store.PrimPath(prim); });
    std::vector<std::uint32_t> depth(slots, 0);
    std::vector<std::uint32_t> parentSlot(slots, kNone);
    for (const PrimId prim : prims) {
        if (IsA(scene_.Schema(prim), Xformable::Definition()) && ResetsXformStack(scene_, prim)) {
            continue;
        }
        for (PathId above = paths.Parent(store.PrimPath(prim)); above != kNoPath;
             above = above == kRootPath ? kNoPath : paths.Parent(above)) {
            if (const std::optional<PrimId> parent = store.Find(above)) {
                parentSlot[prim.slot] = parent->slot;
                depth[prim.slot] = depth[parent->slot] + 1;
                break;
            }
        }
    }
    std::ranges::stable_sort(prims, {}, [&](PrimId prim) { return depth[prim.slot]; });

    const auto count = static_cast<std::uint32_t>(prims.size());
    nodePrim_ = prims;
    slotNode_.assign(slots, kNone);
    levels_.clear();
    for (std::uint32_t node = 0; node < count; ++node) {
        slotNode_[prims[node].slot] = node;
        const std::uint32_t level = depth[prims[node].slot];
        if (level == levels_.size()) {
            levels_.push_back({node, 0});
        }
        ++levels_.back().count;
    }
    std::vector<std::uint32_t> parents(count);
    for (std::uint32_t node = 0; node < count; ++node) {
        const std::uint32_t parent = parentSlot[prims[node].slot];
        parents[node] = parent == kNone ? kNone : slotNode_[parent];
    }

    Queue& queue = context_.Main();
    for (DeviceColumn* column :
         {parents_.get(), locals_.get(), before_.get(), after_.get(), worlds_.get()}) {
        column->Resize(count, queue);
    }
    slotNodes_->Resize(slots, queue);
    uploader_.Write(parents_->Storage(), parents_->Range().offset,
                    std::span<const std::uint32_t>(parents));
    uploader_.Write(slotNodes_->Storage(), slotNodes_->Range().offset,
                    std::span<const std::uint32_t>(slotNode_));
    for (Poses& poses : poses_) {
        const Bucket& bucket = store.GetBucket(poses.bucket);
        std::vector<std::uint32_t> nodes(bucket.Rows(), kNone);
        for (std::uint32_t row = 0; row < bucket.Rows(); ++row) {
            const PrimId prim = bucket.prims[row];
            if (IsA(scene_.Schema(prim), Xformable::Definition()) &&
                SplitAroundPose(scene_, prim)) {
                nodes[row] = slotNode_[prim.slot];
            }
        }
        poses.nodes->Resize(bucket.Rows(), queue);
        uploader_.Write(poses.nodes->Storage(), poses.nodes->Range().offset,
                        std::span<const std::uint32_t>(nodes));
    }
    // Nothing of the old graph is on the device at these nodes: every transform goes up.
    const Rows12 never{std::numeric_limits<float>::quiet_NaN()};
    for (std::vector<Rows12>* uploaded : {&uploadedLocals_, &uploadedBefore_, &uploadedAfter_}) {
        uploaded->assign(count, never);
    }
    for (std::uint32_t node = 0; node < count; ++node) {
        UploadLocal(prims[node], node, FindPoses(store.PrimBucket(prims[node])));
    }
    built_ = true;
}

Ticket TransformGraph::Update(std::span<const Ticket> waits) {
    const PrimStore& store = scene_.Store();
    const ChangeSet changes = store.ChangesSince(cursor_);
    cursor_ = changes.version;
    if (!built_ || !changes.rows.empty()) {
        Rebuild();
    } else {
        // Transforms the host authored: their nodes' locals, or the ops around their poses.
        for (const ChangeSet::Values& values : changes.values) {
            if (!Any(scene_.Classify(values), Dirty::Transform)) {
                continue;
            }
            const Bucket& bucket = store.GetBucket(values.bucket);
            const Poses* poses = FindPoses(values.bucket);
            for (const RowRange& range : values.rows) {
                for (std::uint32_t row = range.begin; row < range.end; ++row) {
                    const PrimId prim = bucket.prims[row];
                    UploadLocal(prim, slotNode_[prim.slot], poses);
                }
            }
        }
    }
    std::vector<Ticket> after(waits.begin(), waits.end());
    after.push_back(uploader_.Flush(waits));
    return context_.Main().Submit(
        [&](VkCommandBuffer commands) {
            // After earlier readers of the worlds and writers of the poses.
            ComputeBarrier(commands);
            for (const Poses& poses : poses_) {
                const auto rows = static_cast<std::uint32_t>(poses.nodes->Rows());
                pose_.Dispatch(commands, Groups(rows),
                               PoseParameters{poses.translate->Address(), poses.orient->Address(),
                                              poses.nodes->Address(), before_->Address(),
                                              after_->Address(), locals_->Address(), rows, 0});
            }
            for (const Level& level : levels_) {
                ComputeBarrier(commands);
                compose_.Dispatch(commands, Groups(level.count),
                                  ComposeParameters{parents_->Address(), locals_->Address(),
                                                    worlds_->Address(), level.first, level.count});
            }
        },
        after);
}

}  // namespace ernie

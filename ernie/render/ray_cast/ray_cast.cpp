// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/ray_cast/ray_cast.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ernie/device/device.h"
#include "ernie/device/program.h"
#include "ernie/device/queue.h"
#include "ernie/render/kernels.h"
#include "ernie/render/kernels/query/ray_cast.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {
namespace {

int64_t CountRays(device::Device& device, const device::Tensor& tensor, int64_t width,
                  device::DType type, const char* what) {
    const Span<const int64_t> shape = tensor.Shape();
    const bool shaped = width == 1 ? shape.size() == 1 : shape.size() == 2 && shape[1] == width;
    const bool typed = tensor.Type() == type ||
                       (type == device::DType::UInt32 && tensor.Type() == device::DType::Int32);
    if (!shaped || !typed) {
        throw Error("Ray {} are a [N{}] {} tensor", what,
                    width == 1 ? "" : ", " + std::to_string(width),
                    type == device::DType::Float32 ? "float32" : "uint32");
    }
    if (&tensor.GetDevice() != &device) {
        throw Error("Ray {} are on {}, not the scene's {}; Device::Copy or Device::Import moves "
                    "them",
                    what, tensor.GetDevice().Adapter().alias, device.Adapter().alias);
    }
    const Span<const int64_t> strides = tensor.Strides();
    if (!strides.empty() && ((shape.back() > 1 && strides.back() != 1) ||
                             (width > 1 && shape[0] > 1 && strides[0] != width))) {
        throw UnsupportedError("Ray {} are cast from contiguous tensors only", what);
    }
    return shape[0];
}

uint64_t InputAddress(device::Device& device, const device::Tensor& tensor, int64_t width,
                      device::DType type, int64_t count, const char* what) {
    if (tensor.Empty()) {
        return 0;
    }
    if (CountRays(device, tensor, width, type, what) != count) {
        throw Error("Ray {} hold {} rows for {} rays", what, tensor.Shape()[0], count);
    }
    return tensor.Address();
}

uint64_t Output(device::Device& device, device::Tensor& tensor, bool wanted, device::DType type,
                std::vector<int64_t> shape) {
    if (!wanted) {
        tensor = device::Tensor();
        return 0;
    }
    const bool fits = !tensor.Empty() && &tensor.GetDevice() == &device && tensor.Type() == type &&
                      std::ranges::equal(tensor.Shape(), shape);
    if (!fits) {
        tensor = device.CreateTensor(type, std::move(shape), "ray hits");
    }
    return tensor.Address();
}

}

struct RayCaster::State {
    Ref<Scene> scene;
    Ref<device::Program> program;
};

RayCaster::RayCaster(Ref<Scene> scene) : m_state(std::make_unique<State>()) {
    if (!scene) {
        throw Error("A ray caster of no scene");
    }
    m_state->scene = std::move(scene);
    m_state->program = m_state->scene->GetDevice().CreateProgram(
        {.modules = {"ernie.render.kernels.query.ray_cast"},
         .library = &kRenderKernels,
         .entries = {"CastClosest", "CastAny"},
         .label = "ray cast",
         .traversal = SceneImpl::Get(*m_state->scene).GetBuilder().Format()});
}

RayCaster::~RayCaster() = default;
RayCaster::RayCaster(RayCaster&&) noexcept = default;
RayCaster& RayCaster::operator=(RayCaster&&) noexcept = default;

device::Operation RayCaster::Cast(const Rays& rays, RayHits& hits, const RayCastDesc& desc,
                                  Stream stream) {
    ERNIE_ZONE("Query/RayCast");
    Scene& scene = *m_state->scene;
    device::Device& device = scene.GetDevice();
    if (rays.origins.Empty() || rays.directions.Empty()) {
        throw Error("A cast needs ray origins and directions");
    }
    constexpr device::DType kFloat = device::DType::Float32;
    const int64_t count = CountRays(device, rays.origins, 3, kFloat, "origins");
    if (CountRays(device, rays.directions, 3, kFloat, "directions") != count) {
        throw Error("{} ray origins with {} directions", count, rays.directions.Shape()[0]);
    }
    if (count > int64_t(~uint32_t(0))) {
        throw Error("A cast of {} rays; at most {} fit one cast", count, ~uint32_t(0));
    }
    RayCastParams params{};
    params.origins.address = InputAddress(device, rays.origins, 3, kFloat, count, "origins");
    params.directions.address =
        InputAddress(device, rays.directions, 3, kFloat, count, "directions");
    params.tMins.address = InputAddress(device, rays.tMins, 1, kFloat, count, "minimum distances");
    params.tMaxes.address =
        InputAddress(device, rays.tMaxes, 1, kFloat, count, "maximum distances");
    params.worlds.address =
        InputAddress(device, rays.worlds, 1, device::DType::UInt32, count, "worlds");
    params.minDistance = rays.minDistance;
    params.maxDistance = rays.maxDistance;
    params.world = rays.world;
    params.count = uint32_t(count);

    const bool closest = desc.mode == RayCastMode::Closest;
    const auto rows = [&](int64_t width) {
        return width == 1 ? std::vector<int64_t>{count} : std::vector<int64_t>{count, width};
    };
    params.hits.address = Output(device, hits.hit, true, device::DType::UInt32, rows(1));
    params.distances.address =
        Output(device, hits.distance, closest && desc.outputs & RayOutput::Distance,
               device::DType::Float32, rows(1));
    params.prims.address = Output(device, hits.prim, closest && desc.outputs & RayOutput::Prim,
                                  device::DType::UInt32, rows(1));
    params.normals.address =
        Output(device, hits.normal, closest && desc.outputs & RayOutput::Normal,
               device::DType::Float32, rows(3));
    params.positions.address =
        Output(device, hits.position, closest && desc.outputs & RayOutput::Position,
               device::DType::Float32, rows(3));

    const device::Operation published = scene.Publish({}, stream);
    if (count == 0) {
        hits.done = published;
        return hits.done;
    }
    const Snapshot snapshot = scene.Acquire();
    params.scene.address = snapshot.sceneData;
    params.tables.address = snapshot.tables;
    device::Queue& queue = device.GetQueue();
    queue.Follow(stream);
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.BeginZone(profiler::ZoneId::Named(desc.label));
    encoder.Dispatch(m_state->program->Entry(closest ? "CastClosest" : "CastAny"), params,
                     {.x = device::GroupCount(params.count, kRayCastGroup)});
    encoder.EndZone();
    hits.done = queue.Submit(encoder.Finish());
    scene.Release(snapshot);
    if (!stream.IsNone()) {
        hits.done.Wait(stream);
    }
    return hits.done;
}

Scene& RayCaster::GetScene() const {
    return *m_state->scene;
}

RayHits RayCaster::Cast(const Rays& rays, const RayCastDesc& desc) {
    RayHits hits;
    Cast(rays, hits, desc, Stream::Host());
    return hits;
}

}

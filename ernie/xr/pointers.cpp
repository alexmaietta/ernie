// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <vector>

#include <dlpack/dlpack.h>

#include "ernie/device/device.h"
#include "ernie/device/stream.h"
#include "ernie/util/profiler.h"
#include "ernie/xr/poses.h"
#include "ernie/xr/rig_state.h"

namespace ernie::xr {
namespace {

Double3 Along(Double3 from, Double3 to, double t) {
    return {from.x + t * (to.x - from.x), from.y + t * (to.y - from.y),
            from.z + t * (to.z - from.z)};
}

double Length(Double3 v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Double3 Forward(const Pose& pose) {
    return Rotate(pose.orientation, {0, 0, -1});
}

device::Tensor Upload(device::Device& device, const std::vector<float>& values, const char* label) {
    int64_t shape[2] = {int64_t(values.size() / 3), 3};
    DLTensor tensor{};
    tensor.data = const_cast<float*>(values.data());
    tensor.device = {kDLCPU, 0};
    tensor.ndim = 2;
    tensor.dtype = {kDLFloat, 32, 1};
    tensor.shape = shape;
    return device.Import(tensor, Stream::None(), label);
}

void Push(std::vector<float>& values, Double3 v) {
    values.insert(values.end(), {float(v.x), float(v.y), float(v.z)});
}

template <class T> std::vector<T> Read(const device::Tensor& tensor) {
    return tensor.Empty() ? std::vector<T>() : tensor.ToHost<T>();
}

}

void RigState::MirrorParent() {
    if (!parentPath) {
        return;
    }
    if (readingParent && parentRead.Ready()) {
        readingParent = false;
        WorldTransform world;
        std::memcpy(&world, parentRead.Bytes().data(), sizeof(world));
        const auto axis = [](Float3 v) {
            const double length =
                std::sqrt(double(v.x) * v.x + double(v.y) * v.y + double(v.z) * v.z);
            return length > 0 ? Double3{v.x / length, v.y / length, v.z / length}
                              : Double3{0, 0, 0};
        };
        parentRotation = FromAxes(axis(world.x), axis(world.y), axis(world.z));
        parentPosition = {double(world.high.x) + world.low.x, double(world.high.y) + world.low.y,
                          double(world.high.z) + world.low.z};
    }
    if (!readingParent && scene->Published() > 0) {
        parentRead = scene->ReadWorldTransformsAsync(std::vector<std::string>{*parentPath});
        readingParent = true;
    }
}

Quatd RigState::WorldRotation() const {
    return Multiply(parentRotation, rootPose.orientation);
}

RayHits RigState::Cast(const std::vector<float>& origins, const std::vector<float>& steps,
                       RayOutput outputs) {
    if (!caster) {
        caster = std::make_unique<RayCaster>(scene);
    }
    device::Device& device = scene->GetDevice();
    Rays rays;
    rays.origins = Upload(device, origins, "xr ray origins");
    rays.directions = Upload(device, steps, "xr ray steps");
    rays.maxDistance = 1;
    rays.space = root;
    RayHits hits;
    caster->Cast(rays, hits,
                 {.outputs = outputs, .label = "XR/Pointers", .ignore = pointers.ignore},
                 Stream::None());
    return hits;
}

PointerBatch RigState::Queue(const Frame& frame, const bool (&kinds)[kPointerKinds][2]) {
    ERNIE_ZONE("XR/Pointer cast");
    PointerBatch batch;
    batch.frame = frame.index;
    std::vector<float> origins;
    std::vector<float> steps;
    const Double3 down = Rotate(Conjugate(WorldRotation()), {0, 0, -1});
    uint32_t count = 0;
    for (uint32_t hand = 0; hand < 2; ++hand) {
        const Controller& controller = frame.controllers[hand];
        batch.active[hand] = controller.active && controller.aim.valid;
        if (!batch.active[hand]) {
            continue;
        }
        const Double3 origin = controller.aim.position;
        const Double3 direction = Forward(controller.aim);
        if (kinds[kLaser][hand]) {
            batch.points[kLaser][hand] = {origin,
                                          Add(origin, Scale(direction, pointers.maxDistance))};
        }
        if (kinds[kTeleport][hand]) {
            const uint32_t segments = std::max(pointers.arcSegments, 1u);
            std::vector<Double3>& points = batch.points[kTeleport][hand];
            for (uint32_t k = 0; k <= segments; ++k) {
                const double t = double(pointers.arcTime) * k / segments;
                points.push_back(Add(Add(origin, Scale(direction, pointers.arcSpeed * t)),
                                     Scale(down, 0.5 * pointers.gravity * t * t)));
            }
        }
        for (uint32_t kind = 0; kind < kPointerKinds; ++kind) {
            const std::vector<Double3>& points = batch.points[kind][hand];
            if (!points.empty()) {
                batch.first[kind][hand] = int32_t(count);
            }
            for (size_t k = 0; k + 1 < points.size(); ++k) {
                Push(origins, points[k]);
                Push(steps, Add(points[k + 1], Scale(points[k], -1)));
                ++count;
            }
        }
    }
    if (count > 0) {
        batch.hits =
            Cast(origins, steps,
                 RayOutput::Distance | RayOutput::Prim | RayOutput::Normal | RayOutput::Position);
        batch.cast = true;
    }
    return batch;
}

void RigState::Resolve(PointerBatch& batch) const {
    if (batch.read) {
        return;
    }
    ERNIE_ZONE("XR/Pointer readback");
    batch.read = true;
    if (batch.cast) {
        batch.hits.done.Wait();
    }
    const std::vector<uint32_t> hit =
        batch.cast ? Read<uint32_t>(batch.hits.hit) : std::vector<uint32_t>();
    const std::vector<float> distance =
        batch.cast ? Read<float>(batch.hits.distance) : std::vector<float>();
    const std::vector<uint32_t> prims =
        batch.cast ? Read<uint32_t>(batch.hits.prim) : std::vector<uint32_t>();
    const std::vector<float> normals =
        batch.cast ? Read<float>(batch.hits.normal) : std::vector<float>();
    const std::vector<float> positions =
        batch.cast ? Read<float>(batch.hits.position) : std::vector<float>();
    const double minimumUp = std::cos(double(pointers.maxSlopeDegrees) * std::numbers::pi / 180);
    for (uint32_t kind = 0; kind < kPointerKinds; ++kind) {
        for (uint32_t hand = 0; hand < 2; ++hand) {
            PointerHit& result = batch.results[kind][hand];
            result = PointerHit();
            result.frame = batch.frame;
            result.active = batch.active[hand];
            const int32_t first = batch.first[kind][hand];
            if (first < 0) {
                continue;
            }
            const std::vector<Double3>& points = batch.points[kind][hand];
            result.path = {points.front()};
            for (size_t k = 0; k + 1 < points.size(); ++k) {
                const size_t i = size_t(first) + k;
                const Double3 from = points[k];
                const Double3 to = points[k + 1];
                if (i < hit.size() && hit[i] != 0) {
                    result.hit = true;
                    result.position = {positions[3 * i], positions[3 * i + 1],
                                       positions[3 * i + 2]};
                    result.normal = {normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]};
                    result.localPosition = Along(from, to, distance[i]);
                    result.prim = prims[i];
                    result.distance += distance[i] * Length(Add(to, Scale(from, -1)));
                    result.path.push_back(result.localPosition);
                    break;
                }
                result.distance += Length(Add(to, Scale(from, -1)));
                result.path.push_back(to);
            }
            if (!result.hit) {
                result.localPosition = result.path.back();
            }
            result.valid = result.hit && (kind == kLaser || result.normal.z >= minimumUp);
        }
    }
}

PointerHit RigState::Pointer(uint32_t kind, Side side) {
    const uint32_t hand = uint32_t(side);
    const bool first = !wanted[kind][hand];
    wanted[kind][hand] = true;
    if (first && lastFrame) {
        bool kinds[kPointerKinds][2] = {};
        kinds[kind][hand] = true;
        PointerBatch alone = Queue(*lastFrame, kinds);
        Resolve(alone);
        return alone.results[kind][hand];
    }
    if (pending && (pointers.wait || pending->Done())) {
        Resolve(*pending);
        if (pending->first[kind][hand] >= 0 || !pending->active[hand]) {
            return pending->results[kind][hand];
        }
    }
    if (latest) {
        Resolve(*latest);
        return latest->results[kind][hand];
    }
    return PointerHit();
}

void Rig::SetPointers(const PointerDesc& desc) {
    m_state->pointers = desc;
}

const PointerDesc& Rig::GetPointers() const {
    return m_state->pointers;
}

PointerHit Rig::Laser(Side hand) {
    return m_state->Pointer(kLaser, hand);
}

PointerHit Rig::Teleport(Side hand) {
    return m_state->Pointer(kTeleport, hand);
}

Pose Rig::WorldPose(const Pose& pose) const {
    const RigState& state = *m_state;
    const Double3 local =
        Add(state.rootPose.position,
            Scale(Rotate(state.rootPose.orientation, pose.position), state.userScale));
    Pose world = pose;
    world.position = Add(state.parentPosition, Rotate(state.parentRotation, local));
    world.orientation = Multiply(state.WorldRotation(), pose.orientation);
    return world;
}

}

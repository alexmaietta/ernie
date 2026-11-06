#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/render/ecs.hpp>

namespace ernie_bench {

using madrona::CountT;
using madrona::Entity;
using madrona::base::ObjectID;
using madrona::base::ObjectInstance;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::math::Diag3x3;
using madrona::math::Quat;
using madrona::math::Vector3;

struct RenderEntity : public madrona::Archetype<ObjectInstance, madrona::render::Renderable> {};

struct CameraEntity : public madrona::Archetype<Position, Rotation, madrona::render::RenderCamera> {
};

}

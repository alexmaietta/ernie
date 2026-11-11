#pragma once

#include <madrona/custom_context.hpp>
#include <madrona/taskgraph_builder.hpp>

#include "types.hpp"

namespace ernie_bench {

class Engine;

enum class ExportID : uint32_t {
    InstancePositions,
    InstanceRotations,
    InstanceScales,
    CameraPositions,
    CameraRotations,
    RaycastDepth,
    RaycastColor,
    NumExports,
};

enum class TaskGraphID : uint32_t {
    Init,
    Render,
    NumGraphs,
};

struct Sim : public madrona::WorldBase {
    struct Config {
        const int32_t* instanceObjects;
        const uint32_t* worldOffsets;
        uint32_t numCameras;
        float fovY;
        float nearPlane;
        const madrona::render::RenderECSBridge* renderBridge;
        bool useRT;
    };

    struct WorldInit {};

    static void registerTypes(madrona::ECSRegistry& registry, const Config& cfg);

    static void setupTasks(madrona::TaskGraphManager& taskgraph_mgr, const Config& cfg);

    Sim(Engine& ctx, const Config& cfg, const WorldInit&);
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
  public:
    using CustomContext::CustomContext;

    template <typename ArchetypeT> inline madrona::Entity makeRenderableEntity();
};

}

#include "sim.inl"

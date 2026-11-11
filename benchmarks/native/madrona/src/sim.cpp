// SPDX-License-Identifier: BSD-3-Clause
#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"

using namespace madrona;
using namespace madrona::math;

namespace RenderingSystem = madrona::render::RenderingSystem;

namespace ernie_bench {

void Sim::registerTypes(ECSRegistry& registry, const Config& cfg) {
    base::registerTypes(registry);
    RenderingSystem::registerTypes(registry, cfg.renderBridge);

    registry.registerArchetype<RenderEntity>();
    registry.exportColumn<RenderEntity, Position>((uint32_t)ExportID::InstancePositions);
    registry.exportColumn<RenderEntity, Rotation>((uint32_t)ExportID::InstanceRotations);
    registry.exportColumn<RenderEntity, Scale>((uint32_t)ExportID::InstanceScales);

    registry.registerArchetype<CameraEntity>();
    registry.exportColumn<CameraEntity, Position>((uint32_t)ExportID::CameraPositions);
    registry.exportColumn<CameraEntity, Rotation>((uint32_t)ExportID::CameraRotations);

    if (cfg.useRT) {
        registry.exportColumn<render::RaycastOutputArchetype, render::DepthOutputBuffer>(
            (uint32_t)ExportID::RaycastDepth);
        registry.exportColumn<render::RaycastOutputArchetype, render::RGBOutputBuffer>(
            (uint32_t)ExportID::RaycastColor);
    }
}

#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
static TaskGraphNodeID sortByWorld(TaskGraphBuilder& builder, Span<const TaskGraphNodeID> deps) {
    auto sorted = builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(deps);
    return builder.addToGraph<ResetTmpAllocNode>({sorted});
}
#endif

void Sim::setupTasks(TaskGraphManager& taskgraph_mgr, const Config&) {
    TaskGraphBuilder& init = taskgraph_mgr.init(TaskGraphID::Init);
#ifdef MADRONA_GPU_MODE
    TaskGraphNodeID sorted = sortByWorld<CameraEntity>(init, {});
    sortByWorld<RenderEntity>(init, {sorted});
#else
    (void)init;
#endif

    TaskGraphBuilder& render = taskgraph_mgr.init(TaskGraphID::Render);
    RenderingSystem::setupTasks(render, {});
}

Sim::Sim(Engine& ctx, const Config& cfg, const WorldInit&) : WorldBase(ctx) {
    RenderingSystem::init(ctx, cfg.renderBridge);

    const uint32_t world = (uint32_t)ctx.worldID().idx;
    for (CountT i = cfg.worldOffsets[world]; i < (CountT)cfg.worldOffsets[world + 1]; i++) {
        Entity instance = ctx.makeRenderableEntity<RenderEntity>();
        ctx.get<Position>(instance) = Vector3::zero();
        ctx.get<Rotation>(instance) = Quat{1, 0, 0, 0};
        ctx.get<Scale>(instance) = Diag3x3{1, 1, 1};
        ctx.get<ObjectID>(instance) = ObjectID{cfg.instanceObjects[i]};
    }

    for (CountT i = 0; i < (CountT)cfg.numCameras; i++) {
        Entity camera = ctx.makeEntity<CameraEntity>();
        ctx.get<Position>(camera) = Vector3::zero();
        ctx.get<Rotation>(camera) = Quat{1, 0, 0, 0};
        RenderingSystem::attachEntityToView(ctx, camera, cfg.fovY, cfg.nearPlane, Vector3::zero());
    }
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}

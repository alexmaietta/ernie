// SPDX-License-Identifier: BSD-3-Clause
#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/cuda_utils.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/importer.hpp>
#include <madrona/mw_gpu.hpp>
#include <madrona/render/api.hpp>
#include <madrona/render/asset_processor.hpp>
#include <madrona/render/render_mgr.hpp>

#include <cstring>

using namespace madrona;
using namespace madrona::math;

namespace ernie_bench {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};

struct Manager::Impl {
    Config cfg;
    uint32_t instanceCount;
    uint32_t cameraCount;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph renderGraph;
    Optional<MWCudaLaunchGraph> raytraceGraph;
    cudaStream_t stream;

    void copyIn(ExportID slot, const void* host, size_t bytes) {
        REQ_CUDA(
            cudaMemcpy(gpuExec.getExported((uint32_t)slot), host, bytes, cudaMemcpyHostToDevice));
    }

    size_t pixelCount() const {
        return (size_t)cfg.worldCount * cameraCount * cfg.width * cfg.height;
    }
};

static HeapArray<imp::SourceMesh> sourceMeshes(const MeshBuilder& geometry) {
    HeapArray<imp::SourceMesh> meshes(geometry.meshCount);
    for (uint32_t mesh = 0; mesh < geometry.meshCount; mesh++) {
        uint32_t first_vertex = geometry.vertexOffsets[mesh];
        uint32_t end_vertex =
            mesh + 1 < geometry.meshCount ? geometry.vertexOffsets[mesh + 1] : geometry.vertexCount;
        uint32_t first_triangle = geometry.triangleOffsets[mesh];
        uint32_t end_triangle = mesh + 1 < geometry.meshCount ? geometry.triangleOffsets[mesh + 1]
                                                              : geometry.triangleCount;
        meshes[mesh] = {
            .positions = (Vector3*)(geometry.vertices + 3 * first_vertex),
            .normals = (Vector3*)(geometry.normals + 3 * first_vertex),
            .tangentAndSigns = nullptr,
            .uvs = nullptr,
            .indices = (uint32_t*)(geometry.indices + 3 * first_triangle),
            .faceCounts = nullptr,
            .faceMaterials = nullptr,
            .numVertices = end_vertex - first_vertex,
            .numFaces = end_triangle - first_triangle,
            .materialIDX = 0,
        };
    }
    return meshes;
}

Manager::Manager(const Config& cfg, const MeshBuilder& geometry, const int32_t* instance_objects,
                 uint32_t instance_count, uint32_t camera_count) {
    HeapArray<imp::SourceMesh> meshes = sourceMeshes(geometry);
    HeapArray<imp::SourceMesh> object_meshes(geometry.objectCount);
    HeapArray<imp::SourceObject> objects(geometry.objectCount);
    HeapArray<imp::SourceMaterial> materials(geometry.objectCount);
    for (uint32_t object = 0; object < geometry.objectCount; object++) {
        object_meshes[object] = meshes[geometry.objectMeshes[object]];
        object_meshes[object].materialIDX = object;
        objects[object] = {.meshes = Span<imp::SourceMesh>(&object_meshes[object], 1)};
        const float* color = geometry.objectColors + 3 * object;
        materials[object] = {Vector4{color[0], color[1], color[2], 1.f}, -1, 0.8f, 0.2f};
    }

    CUcontext cu_ctx = MWCudaExecutor::initCUDA(cfg.gpuID);

    Optional<RenderGPUState> render_gpu_state = Optional<RenderGPUState>::none();
    Optional<render::RenderManager> render_mgr = Optional<render::RenderManager>::none();
    if (!cfg.useRT) {
        auto api_lib = render::APIManager::loadDefaultLib();
        render::APIManager api_mgr(api_lib.lib());
        render::GPUHandle gpu = api_mgr.initGPU(cfg.gpuID);
        render_gpu_state = RenderGPUState{std::move(api_lib), std::move(api_mgr), std::move(gpu)};
        render_mgr.emplace(render_gpu_state->apiMgr.backend(), render_gpu_state->gpu.device(),
                           render::RenderManager::Config{
                               .enableBatchRenderer = true,
                               .renderMode = render::RenderManager::Config::RenderMode::RGBD,
                               .agentViewWidth = cfg.width,
                               .agentViewHeight = cfg.height,
                               .numWorlds = cfg.worldCount,
                               .maxViewsPerWorld = camera_count,
                               .maxInstancesPerWorld = instance_count,
                               .execMode = ExecMode::CUDA,
                               .voxelCfg = {},
                           });
        render_mgr->loadObjects(objects, materials, {});
        Vector3 light{cfg.lightDirection[0], cfg.lightDirection[1], cfg.lightDirection[2]};
        render_mgr->configureLighting({{true, light, Vector3{1.f, 1.f, 1.f}}});
    }

    int32_t* instance_objects_gpu = (int32_t*)cu::allocGPU(sizeof(int32_t) * instance_count);
    REQ_CUDA(cudaMemcpy(instance_objects_gpu, instance_objects, sizeof(int32_t) * instance_count,
                        cudaMemcpyHostToDevice));

    Sim::Config sim_cfg{
        .instanceObjects = instance_objects_gpu,
        .numInstances = instance_count,
        .numCameras = camera_count,
        .fovY = cfg.fovY,
        .nearPlane = cfg.nearPlane,
        .renderBridge = render_mgr.has_value() ? render_mgr->bridge() : nullptr,
        .useRT = cfg.useRT,
    };

    Optional<CudaBatchRenderConfig> raytrace_cfg = Optional<CudaBatchRenderConfig>::none();
    if (cfg.useRT) {
        raytrace_cfg = CudaBatchRenderConfig{
            .renderMode = CudaBatchRenderConfig::RenderMode::RGBD,
            .geoBVHData = render::AssetProcessor::makeBVHData(objects),
            .materialData = render::AssetProcessor::initMaterialData(materials.data(),
                                                                     materials.size(), nullptr, 0),
            .renderResolution = cfg.width,
            .nearPlane = cfg.nearPlane,
            .farPlane = cfg.farPlane,
        };
    }

    HeapArray<Sim::WorldInit> world_inits(cfg.worldCount);
    MWCudaExecutor gpu_exec(
        {
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(Sim::WorldInit),
            .userConfigPtr = (void*)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = cfg.worldCount,
            .numTaskGraphs = (uint32_t)TaskGraphID::NumGraphs,
            .numExportedBuffers = (uint32_t)ExportID::NumExports,
        },
        {
            {GPU_SIM_SRC_LIST},
            {GPU_SIM_COMPILE_FLAGS},
            CompileConfig::OptMode::LTO,
        },
        cu_ctx, raytrace_cfg);

    Optional<MWCudaLaunchGraph> raytrace_graph = Optional<MWCudaLaunchGraph>::none();
    if (cfg.useRT) {
        raytrace_graph = gpu_exec.buildRenderGraph();
    }
    cu::deallocGPU(instance_objects_gpu);

    MWCudaLaunchGraph render_graph = gpu_exec.buildLaunchGraph(TaskGraphID::Render);
    MWCudaLaunchGraph init_graph = gpu_exec.buildLaunchGraph(TaskGraphID::Init);
    gpu_exec.run(init_graph);

    cudaStream_t stream;
    REQ_CUDA(cudaStreamCreate(&stream));

    impl_ = std::unique_ptr<Impl>(new Impl{
        cfg,
        instance_count,
        camera_count,
        std::move(render_gpu_state),
        std::move(render_mgr),
        std::move(gpu_exec),
        std::move(render_graph),
        std::move(raytrace_graph),
        stream,
    });
}

Manager::~Manager() {
    REQ_CUDA(cudaStreamSynchronize(impl_->stream));
    REQ_CUDA(cudaStreamDestroy(impl_->stream));
}

void Manager::setInstances(const float* positions, const float* rotations, const float* scales) {
    size_t count = (size_t)impl_->cfg.worldCount * impl_->instanceCount;
    impl_->copyIn(ExportID::InstancePositions, positions, sizeof(Vector3) * count);
    impl_->copyIn(ExportID::InstanceRotations, rotations, sizeof(Quat) * count);
    impl_->copyIn(ExportID::InstanceScales, scales, sizeof(Diag3x3) * count);
}

void Manager::setCameras(const float* positions, const float* rotations) {
    size_t count = (size_t)impl_->cfg.worldCount * impl_->cameraCount;
    impl_->copyIn(ExportID::CameraPositions, positions, sizeof(Vector3) * count);
    impl_->copyIn(ExportID::CameraRotations, rotations, sizeof(Quat) * count);
}

void Manager::render() {
    if (impl_->renderMgr.has_value()) {
        impl_->gpuExec.run(impl_->renderGraph);
        impl_->renderMgr->readECS();
        impl_->renderMgr->batchRender();
        return;
    }
    impl_->gpuExec.runAsync(impl_->renderGraph, impl_->stream);
    impl_->gpuExec.runAsync(*impl_->raytraceGraph, impl_->stream);
}

void Manager::synchronize() {
    REQ_CUDA(cudaStreamSynchronize(impl_->stream));
}

uintptr_t Manager::stream() const {
    return impl_->renderMgr.has_value() ? 0 : (uintptr_t)impl_->stream;
}

void Manager::readColor(uint8_t* out) const {
    const void* source = impl_->renderMgr.has_value()
                             ? (const void*)impl_->renderMgr->batchRendererRGBOut()
                             : impl_->gpuExec.getExported((uint32_t)ExportID::RaycastColor);
    REQ_CUDA(cudaMemcpy(out, source, 4 * impl_->pixelCount(), cudaMemcpyDeviceToHost));
}

void Manager::readDepth(float* out) const {
    const void* source = impl_->renderMgr.has_value()
                             ? (const void*)impl_->renderMgr->batchRendererDepthOut()
                             : impl_->gpuExec.getExported((uint32_t)ExportID::RaycastDepth);
    REQ_CUDA(cudaMemcpy(out, source, sizeof(float) * impl_->pixelCount(), cudaMemcpyDeviceToHost));
}

uint32_t Manager::worldCount() const {
    return impl_->cfg.worldCount;
}

uint32_t Manager::instanceCount() const {
    return impl_->instanceCount;
}

uint32_t Manager::cameraCount() const {
    return impl_->cameraCount;
}

uint64_t Manager::pixelCount() const {
    return impl_->pixelCount();
}

}

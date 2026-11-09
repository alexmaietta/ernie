// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/bvh/builder.h"
#include "ernie/device/device.h"
#include "ernie/device/ref.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tables/bounds.h"
#include "ernie/scene/tables/captures.h"
#include "ernie/scene/tables/instances.h"
#include "ernie/scene/tables/lights.h"
#include "ernie/scene/tables/materials.h"
#include "ernie/scene/tables/meshes.h"
#include "ernie/scene/tables/pipelines.h"
#include "ernie/scene/tables/plans.h"
#include "ernie/scene/tables/points.h"
#include "ernie/scene/tables/products.h"
#include "ernie/scene/tables/semantics.h"
#include "ernie/scene/tables/sensors.h"
#include "ernie/scene/tables/spectra.h"
#include "ernie/scene/tables/subsets.h"
#include "ernie/scene/tables/surfaces.h"
#include "ernie/scene/tables/table.h"
#include "ernie/scene/tables/tangents.h"
#include "ernie/scene/tables/textures.h"
#include "ernie/scene/tables/transforms.h"
#include "ernie/stage/columns/column_store.h"
#include "ernie/stage/publish/publisher.h"
#include "ernie/stage/structure/structure.h"

namespace ernie {

class SceneImpl {
  public:
    SceneImpl(Scene& scene, device::Device& device, const SceneDesc& desc);

    static SceneImpl& Get(Scene& scene) {
        return *scene.m_impl;
    }

    static const SceneImpl& Get(const Scene& scene) {
        return *scene.m_impl;
    }

    template <class Write> Prim Define(std::string_view path, SchemaId schema, const Write& write) {
        return Define(path, schema, 0, write);
    }

    template <class Write>
    Prim Define(std::string_view path, SchemaId schema, uint64_t apis, const Write& write) {
        const std::string paths[] = {std::string(path)};
        return Handle(m_structure.Define(schema, paths, apis, write).firstPrim);
    }

    template <class Write>
    PrimRange Define(Span<const std::string> paths, SchemaId schema, const Write& write,
                     const BatchDesc& batch = {}) {
        return Define(paths, schema, 0, write, batch);
    }

    template <class Write>
    PrimRange Define(Span<const std::string> paths, SchemaId schema, uint64_t apis,
                     const Write& write, const BatchDesc& batch = {}) {
        return Range(m_structure.Define(schema, std::span(paths.data(), paths.size()), apis, write,
                                        batch.attachable));
    }

    template <class Write> void Apply(Span<const Prim> prims, SchemaId api, const Write& write) {
        m_structure.Apply(Indices(prims), api, write);
    }

    template <class Write>
    void Apply(Span<const Prim> prims, SchemaId api, std::string_view instance,
               const Write& write) {
        m_structure.Apply(Indices(prims), api, instance, write);
    }

    std::vector<uint32_t> Indices(Span<const Prim> prims) const;

    Prim Handle(uint32_t prim) const;
    PrimRange Range(const PrimBlock& block) const;
    void Check(const Prim& prim) const;

    device::Device& GetDevice() const {
        return *m_device;
    }

    ColumnStore& GetColumnStore() {
        return m_store;
    }

    Structure& GetStructure() {
        return m_structure;
    }

    const Structure& GetStructure() const {
        return m_structure;
    }

    Publisher& GetPublisher() {
        return m_publisher;
    }

    bvh::Builder& GetBuilder() {
        return *m_builder;
    }

    TransformTable& GetTransforms() {
        return m_transforms;
    }

    MeshTable& GetMeshes() {
        return m_meshes;
    }

    PointsTable& GetPoints() {
        return m_points;
    }

    ernie::BoundsTable& GetBounds() {
        return m_bounds;
    }

    TangentTable& GetTangents() {
        return m_tangents;
    }

    TextureTable& GetTextures() {
        return m_textures;
    }

    const MaterialTable& GetMaterials() const {
        return m_materials;
    }

    bool WaitsForKernels() const {
        return m_waitForKernels;
    }

    const SubsetTable& GetSubsets() const {
        return m_subsets;
    }

    SemanticTable& GetSemantics() {
        return m_semantics;
    }

    InstanceTable& GetInstances() {
        return m_instances;
    }

    LightTable& GetLights() {
        return m_lights;
    }

    SurfaceTable& GetSurfaces() {
        return m_surfaces;
    }

    SimulationClock& GetClock() {
        return m_clock;
    }

    ProductTable& GetProducts() {
        return m_products;
    }

    const PipelineTable& GetPipelines() const {
        return m_pipelines;
    }

    ProductPlans& GetPlans() {
        return m_plans;
    }

    CaptureSchedule& GetCaptures() {
        return m_captures;
    }

    void AddCommitStep(std::function<bool()> step) {
        m_commitSteps.push_back(std::move(step));
    }

    void RunCommitSteps() {
        std::erase_if(m_commitSteps, [](const std::function<bool()>& step) { return step(); });
    }

    bool CommitStepsPending() const {
        return !m_commitSteps.empty();
    }

  private:
    Scene& m_scene;
    Ref<device::Device> m_device;
    bool m_waitForKernels = true;
    ColumnStore m_store;
    Structure m_structure;
    RenderHeader m_header;
    Publisher m_publisher;
    std::unique_ptr<bvh::Builder> m_builder;
    TransformTable m_transforms;
    MeshTable m_meshes;
    PointsTable m_points;
    ernie::BoundsTable m_bounds;
    TangentTable m_tangents;
    TextureTable m_textures;
    SemanticTable m_semantics;
    SubsetTable m_subsets;
    SensorTable m_sensors;
    SimulationClock m_clock;
    PipelineTable m_pipelines;
    ProductTable m_products;
    ProductPlans m_plans;
    CaptureSchedule m_captures;
    MaterialTable m_materials;
    SpectralTable m_spectra;
    InstanceTable m_instances;
    SurfaceTable m_surfaces;
    LightTable m_lights;
    std::vector<std::function<bool()>> m_commitSteps;
};

}

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
#include "ernie/scene/builders/bounds.h"
#include "ernie/scene/builders/instances.h"
#include "ernie/scene/builders/lights.h"
#include "ernie/scene/builders/materials.h"
#include "ernie/scene/builders/meshes.h"
#include "ernie/scene/builders/products.h"
#include "ernie/scene/builders/semantics.h"
#include "ernie/scene/builders/sensors.h"
#include "ernie/scene/builders/surfaces.h"
#include "ernie/scene/builders/tangents.h"
#include "ernie/scene/builders/textures.h"
#include "ernie/scene/builders/transforms.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/publish/publisher.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/structure/structure.h"

namespace ernie {

void SetInputs(RowWriter& writer, SchemaId schema, std::span<const MaterialInput> inputs);

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

    TransformBuilder& GetTransforms() {
        return m_transforms;
    }

    MeshBuilder& GetMeshes() {
        return m_meshes;
    }

    ernie::BoundsBuilder& GetBounds() {
        return m_bounds;
    }

    TangentBuilder& GetTangents() {
        return m_tangents;
    }

    TextureBuilder& GetTextures() {
        return m_textures;
    }

    SemanticBuilder& GetSemantics() {
        return m_semantics;
    }

    InstanceBuilder& GetInstances() {
        return m_instances;
    }

    LightBuilder& GetLights() {
        return m_lights;
    }

    SurfaceBuilder& GetSurfaces() {
        return m_surfaces;
    }

    ProductBuilder& GetProducts() {
        return m_products;
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
    ColumnStore m_store;
    Structure m_structure;
    Publisher m_publisher;
    std::unique_ptr<bvh::Builder> m_builder;
    TransformBuilder m_transforms;
    MeshBuilder m_meshes;
    ernie::BoundsBuilder m_bounds;
    TangentBuilder m_tangents;
    TextureBuilder m_textures;
    SemanticBuilder m_semantics;
    SensorBuilder m_sensors;
    ProductBuilder m_products;
    MaterialBuilder m_materials;
    InstanceBuilder m_instances;
    SurfaceBuilder m_surfaces;
    LightBuilder m_lights;
    std::vector<std::function<bool()>> m_commitSteps;
};

}

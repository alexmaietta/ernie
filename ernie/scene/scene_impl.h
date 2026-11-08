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
#include "ernie/scene/catalog/catalog.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/store/store.h"
#include "ernie/scene/systems/bounds.h"
#include "ernie/scene/systems/geometry.h"
#include "ernie/scene/systems/hierarchy.h"
#include "ernie/scene/systems/instances.h"
#include "ernie/scene/systems/lights.h"
#include "ernie/scene/systems/readback.h"
#include "ernie/scene/systems/semantics.h"
#include "ernie/scene/systems/surfaces.h"
#include "ernie/scene/systems/tangents.h"
#include "ernie/scene/systems/textures.h"
#include "ernie/scene/timeline/timeline.h"

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
        return Handle(m_catalog.Define(schema, paths, apis, write).firstPrim);
    }

    template <class Write>
    PrimRange Define(Span<const std::string> paths, SchemaId schema, const Write& write,
                     const BatchDesc& batch = {}) {
        return Define(paths, schema, 0, write, batch);
    }

    template <class Write>
    PrimRange Define(Span<const std::string> paths, SchemaId schema, uint64_t apis,
                     const Write& write, const BatchDesc& batch = {}) {
        return Range(m_catalog.Define(schema, std::span(paths.data(), paths.size()), apis, write,
                                      batch.attachable));
    }

    template <class Write> void Apply(Span<const Prim> prims, SchemaId api, const Write& write) {
        m_catalog.Apply(Indices(prims), api, write);
    }

    template <class Write>
    void Apply(Span<const Prim> prims, SchemaId api, std::string_view instance,
               const Write& write) {
        m_catalog.Apply(Indices(prims), api, instance, write);
    }

    std::vector<uint32_t> Indices(Span<const Prim> prims) const;

    Prim Handle(uint32_t prim) const;
    PrimRange Range(const PrimBlock& block) const;
    void Check(const Prim& prim) const;

    device::Device& GetDevice() const {
        return *m_device;
    }

    Store& GetStore() {
        return m_store;
    }

    Catalog& GetCatalog() {
        return m_catalog;
    }

    const Catalog& GetCatalog() const {
        return m_catalog;
    }

    Timeline& GetTimeline() {
        return m_timeline;
    }

    bvh::Builder& GetBuilder() {
        return *m_builder;
    }

    Hierarchy& GetHierarchy() {
        return m_hierarchy;
    }

    Geometry& GetGeometry() {
        return m_geometry;
    }

    ernie::Bounds& GetBounds() {
        return m_bounds;
    }

    Tangents& GetTangents() {
        return m_tangents;
    }

    Textures& GetTextures() {
        return m_textures;
    }

    Semantics& GetSemantics() {
        return m_semantics;
    }

    Instances& GetInstances() {
        return m_instances;
    }

    Lights& GetLights() {
        return m_lights;
    }

    Surfaces& GetSurfaces() {
        return m_surfaces;
    }

    Readback& GetReadback() {
        return m_readback;
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
    Store m_store;
    Catalog m_catalog;
    Timeline m_timeline;
    std::unique_ptr<bvh::Builder> m_builder;
    Hierarchy m_hierarchy;
    Geometry m_geometry;
    ernie::Bounds m_bounds;
    Tangents m_tangents;
    Textures m_textures;
    Semantics m_semantics;
    Instances m_instances;
    Surfaces m_surfaces;
    Lights m_lights;
    Readback m_readback;
    std::vector<std::function<bool()>> m_commitSteps;
};

}

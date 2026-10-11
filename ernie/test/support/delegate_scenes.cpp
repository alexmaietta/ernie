// SPDX-License-Identifier: BSD-3-Clause
#include "support/delegate_scenes.h"

#include "ernie/raster/render_targets.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string_view>

namespace ernie::testing {

Tensor HeightsOf(const TestRaster& raster) {
    auto owner = std::make_shared<std::vector<float>>(raster.values);
    return Tensor(owner->data(), DType::Float32,
                  {static_cast<std::int64_t>(raster.height),
                   static_cast<std::int64_t>(raster.width)},
                  {}, {}, owner);
}

Scene TerrainScene(const TestRaster& raster) {
    Scene scene;
    Terrain::Define(scene, "/World/Terrain")
        .SetHeights(HeightsOf(raster))
        .SetOrigin(glm::dvec2(raster.originX, raster.originY))
        .SetCellSize(raster.cellSize)
        .SetMaterialBinding("/Looks/Rock");
    Material::Define(scene, "/Looks/Rock").SetAlbedo(glm::dvec3(0.6));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform,
             DistantLightTransform({1.0, 1.0, 0.6}));
    RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kSize, kSize))
        .SetClearColor(glm::dvec3(0.0, 0.0, 1.0));
    return scene;
}

void DefineQuad(Scene& scene, const Path& path, float half) {
    const std::array<float, 12> points{-half, -half, 0,    half,
                                       -half, 0,     half, half,
                                       0,     -half, half, 0};
    const std::array<std::int64_t, 1> counts{4};
    const std::array<std::int64_t, 4> indices{0, 1, 2, 3};
    Mesh::Define(scene, path)
        .SetPoints(Tensor::Copy<float>(points, {4, 3}))
        .SetFaceVertexCounts(Tensor::Copy<std::int64_t>(counts, {1}))
        .SetFaceVertexIndices(
            Tensor::Copy<std::int64_t>(indices, {4}));
}

Scene MeshScene() {
    Scene scene;
    Camera::Define(scene, "/World/Camera").SetFocalLength(18.0);
    scene.Set("/World/Camera", tokens::kTransform,
              LookAtTransform({0.0, 0.0, 60.0}, {0.0, 0.0, 0.0},
                              {0.0, 1.0, 0.0}));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform,
             DistantLightTransform({0.0, 0.0, 1.0}));
    RenderSettings::Define(scene, "/Render/Settings")
        .SetResolution(glm::dvec2(kSize, kSize))
        .SetAmbient(0.0);
    Material::Define(scene, "/Looks/Red")
        .SetAlbedo(glm::dvec3(1.0, 0.0, 0.0));
    DefineQuad(scene, "/World/Quad");
    Mesh(scene, "/World/Quad").SetMaterialBinding("/Looks/Red");
    scene.Set(
        "/World/Quad", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(-12.0, 0.0, 0.0)));

    const std::array<float, 9> positions{12, -10, 0,  12, 0,
                                         0,  12,  10, 0};
    PointInstancer::Define(scene, "/World/Rocks")
        .SetPositions(Tensor::Copy<float>(positions, {3, 3}));
    scene.Set("/World/Rocks", tokens::kPrototypes,
              std::vector<Path>{"/World/Rocks/Rock"});
    DefineQuad(scene, "/World/Rocks/Rock", 1.5f);
    Mesh(scene, "/World/Rocks/Rock")
        .SetDisplayColor(glm::dvec3(0.0, 1.0, 0.0));
    return scene;
}

Scene CullingScene() {
    Scene scene = MeshScene();
    std::vector<float> positions;
    for (int i = -20; i <= 20; ++i) {
        positions.insert(
            positions.end(),
            {5.0f * static_cast<float>(i), -20.0f, 0.0f});
    }
    PointInstancer::Define(scene, "/World/Row")
        .SetPositions(Tensor::Copy<float>(positions, {41, 3}));
    scene.Set("/World/Row", tokens::kPrototypes,
              std::vector<Path>{"/World/Rocks/Rock"});
    return scene;
}

const std::uint8_t* TexelAt(const Scene& scene, const Tensor& image,
                            glm::vec3 p) {
    const glm::vec4 clip =
        CameraParamsOf(scene, "/World/Camera", kSize, kSize)
            .RasterView()
            .ViewProjection() *
        glm::vec4(p, 1.0f);
    const glm::vec2 ndc = glm::vec2(clip) / clip.w;
    const auto col =
        static_cast<std::size_t>((ndc.x * 0.5f + 0.5f) * kSize);
    const auto row =
        static_cast<std::size_t>((0.5f - ndc.y * 0.5f) * kSize);
    return image.As<std::uint8_t>() + 4 * (row * kSize + col);
}

Scene FlatScene() {
    Scene scene;
    Terrain::Define(scene, "/World/Terrain")
        .SetHeights(HeightsOf(TestRaster::Centred(65, 65, 2.0)))
        .SetOrigin(glm::dvec2(-65.0, 65.0))
        .SetCellSize(2.0)
        .SetMaterialBinding("/Looks/Grey");
    Material::Define(scene, "/Looks/Grey").SetAlbedo(glm::dvec3(0.5));
    DistantLight::Define(scene, "/World/Sun").SetShadowEnable(false);
    scene.Set("/World/Sun", tokens::kTransform,
              DistantLightTransform({0.0, 0.0, 1.0}));
    Camera::Define(scene, "/World/Camera").SetSensorNoise(false);
    scene.Set("/World/Camera", tokens::kTransform,
              LookAtTransform({0.0, 0.0, 100.0}, {0.0, 0.0, 0.0},
                              {0.0, 1.0, 0.0}));
    std::vector<Path> vars;
    for (const std::string_view name : RenderTargets::VarNames()) {
        const Path var =
            Path("/Render/Vars").Child(std::string(name));
        RenderVar::Define(scene, var).SetSourceName(name);
        vars.push_back(var);
    }
    RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kVarSize, kVarSize))
        .SetColorMode(tokens::kHillshade)
        .SetAmbient(0.0);
    RenderProduct::Define(scene, "/Render/Product");
    scene.Set("/Render/Product", tokens::kOrderedVars, vars);
    scene.Set("/Render/Settings", tokens::kProducts,
              std::vector<Path>{"/Render/Product"});
    return scene;
}

std::map<std::string, Tensor, std::less<>>
MapVars(RenderProductSetOutputs& outputs) {
    std::map<std::string, Tensor, std::less<>> tensors;
    for (auto& [name, var] : outputs.at("/Render/Product").vars) {
        tensors[name] = var.Map(MapDevice::Cpu).tensor;
    }
    return tensors;
}

std::pair<std::size_t, std::size_t> PixelOf(const Scene& scene,
                                            float x, float y) {
    const glm::vec4 clip =
        CameraParamsOf(scene, "/World/Camera", kSize, kSize)
            .RasterView()
            .ViewProjection() *
        glm::vec4(x, y, 0.0f, 1.0f);
    const glm::vec2 ndc = glm::vec2(clip) / clip.w;
    return {
        static_cast<std::size_t>((0.5f - ndc.y * 0.5f) * kVarSize),
        static_cast<std::size_t>((ndc.x * 0.5f + 0.5f) * kVarSize)};
}

std::vector<std::uint8_t> Bytes(const Tensor& image) {
    const auto bytes = static_cast<std::size_t>(image.Elements()) *
                       (image.Type() == DType::UInt8 ? 1 : 4);
    const auto* data = static_cast<const std::uint8_t*>(image.Data());
    return {data, data + bytes};
}

std::size_t BytesDiffering(const std::vector<std::uint8_t>& a,
                           const std::vector<std::uint8_t>& b) {
    std::size_t differing = 0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        differing += std::abs(int(a[i]) - int(b[i])) > 8 ? 1u : 0u;
    }
    return differing + (a.size() > b.size() ? a.size() - b.size()
                                            : b.size() - a.size());
}

}  // namespace ernie::testing

// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/offscreen_target.h"
#include "ernie/geometry/primitives.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/orbit_controller.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/renderer.h"
#include "support/delegate_scenes.h"
#include "support/test_raster.h"
#include "support/vulkan_fixture.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ernie::DeviceKind;
using ernie::DType;
using ernie::OrbitController;
using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

TEST_F(RasterDelegateTest, MeshesOutsideAViewAreCulled) {
    Scene scene = MeshScene();
    RasterDelegate delegate(*context_);
    const Tensor before = delegate.Render(scene);
    const std::size_t drawn = delegate.Meshes().DrawCount(0);
    EXPECT_EQ(drawn, 2u) << "the quad and the rocks";

    // A quad above the camera, which looks down, and rocks far off to
    // the side.
    DefineQuad(scene, "/World/Behind", 2.0f);
    scene.Set(
        "/World/Behind", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 100.0)));
    ernie::PointInstancer::Define(scene, "/World/FarRocks")
        .SetPositions(Tensor::Copy<float>(
            std::vector<float>{500, 0, 0, 600, 0, 0}, {2, 3}));
    scene.Set("/World/FarRocks", tokens::kPrototypes,
              std::vector<Path>{"/World/Rocks/Rock"});
    const Tensor after = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().DrawCount(0), drawn)
        << "neither new draw is in view";
    EXPECT_EQ(Bytes(after), Bytes(before));
    bool counted = false;
    for (const auto& [name, value] : delegate.Stats().counters) {
        if (name == ernie::raster_stats::kMeshDraws) {
            EXPECT_EQ(value, drawn);
            counted = true;
        }
    }
    EXPECT_TRUE(counted);
}

namespace {

/// How many of the row's 41 instances lie in the view of MeshScene's
/// camera, by projecting their centres: the ones GPU culling must
/// keep (the prototype is small next to their spacing).
std::uint32_t RowInstancesInView(const Scene& scene) {
    const glm::mat4 viewProjection =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize)
            .RasterView()
            .ViewProjection();
    std::uint32_t inside = 0;
    for (int i = -20; i <= 20; ++i) {
        const glm::vec4 clip =
            viewProjection * glm::vec4(5.0f * static_cast<float>(i),
                                       -20.0f, 0.0f, 1.0f);
        inside +=
            std::abs(clip.x) <= clip.w && std::abs(clip.y) <= clip.w
                ? 1u
                : 0u;
    }
    return inside;
}

}  // namespace

TEST(CullingOf, ReadsTheSettingsToken) {
    Scene scene;
    scene.Define("/Render/Settings", tokens::kRenderSettings);
    EXPECT_EQ(ernie::CullingOf(scene, "/Render/Settings"),
              ernie::Culling::Gpu)
        << "the default";
    for (const auto& [token, mode] :
         {std::pair{tokens::kOff, ernie::Culling::Off},
          std::pair{tokens::kCpu, ernie::Culling::Cpu},
          std::pair{tokens::kGpu, ernie::Culling::Gpu}}) {
        scene.Set("/Render/Settings", tokens::kCulling,
                  std::string(token));
        EXPECT_EQ(ernie::CullingOf(scene, "/Render/Settings"), mode);
    }
    scene.Set("/Render/Settings", tokens::kCulling,
              std::string("magic"));
    EXPECT_THROW(ernie::CullingOf(scene, "/Render/Settings"),
                 std::invalid_argument);
}

TEST_F(RasterDelegateTest, EveryCullingModeRendersTheSameImage) {
    Scene scene = CullingScene();
    RasterDelegate delegate(*context_);
    std::map<std::string, std::vector<std::uint8_t>> images;
    std::map<std::string, std::size_t> draws;
    for (const std::string_view mode :
         {tokens::kOff, tokens::kCpu, tokens::kGpu}) {
        ernie::RenderSettings(scene, "/Render/Settings")
            .SetCulling(mode);
        images[std::string(mode)] = Bytes(delegate.Render(scene));
        draws[std::string(mode)] = delegate.Meshes().DrawCount(0);
    }
    EXPECT_EQ(images["cpu"], images["off"]);
    EXPECT_EQ(images["gpu"], images["off"]);
    EXPECT_EQ(draws["off"], draws["cpu"])
        << "every draw has some instance in view";
}

TEST_F(RasterDelegateTest, GpuCullingKeepsExactlyTheInstancesInView) {
    Scene scene = CullingScene();
    ernie::DistantLight(scene, "/World/Sun")
        .SetShadowEnable(false);  // only the view's jobs
    RasterDelegate delegate(*context_);
    delegate.Render(scene);
    // Jobs in order: the quad is no instancer; then the rocks (3
    // instances, all in view), then the row.
    const std::vector<std::uint32_t> kept =
        delegate.Meshes().CulledInstanceCounts();
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_EQ(kept[0], 3u);
    const std::uint32_t inView = RowInstancesInView(scene);
    EXPECT_GT(inView, 0u);
    EXPECT_LT(inView, 41u);
    EXPECT_EQ(kept[1], inView);
    EXPECT_EQ(delegate.Meshes().CulledLevelCounts().front(),
              std::vector<std::uint32_t>{1u})
        << "the quad culled on the GPU too, a job of its one "
           "instance";
}

TEST_F(RasterDelegateTest,
       InstancesWrittenAsCudaArraysAreCulledOnTheGpu) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene scene = CullingScene();
    ernie::DistantLight(scene, "/World/Sun").SetShadowEnable(false);
    // With an extent, the positions never come to the host: only the
    // GPU sees each instance.
    scene.Set("/World/Row", tokens::kExtent,
              std::vector<float>{-101.0f, -21.0f, -1.0f, 101.0f,
                                 -19.0f, 1.0f});
    scene.Write(
        "/World/Row", tokens::kPositions,
        cuda->Upload(*scene.Read("/World/Row", tokens::kPositions)));
    delegate.Render(scene);
    const std::vector<std::uint32_t> kept =
        delegate.Meshes().CulledInstanceCounts();
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_EQ(kept[1], RowInstancesInView(scene));
}

namespace {

/// Spheres of radius 1 in a row receding from a camera at the origin
/// looking along +y, every 4 m from 6 m to 400 m, lit from above
/// without shadows.
Scene RecedingSpheres() {
    Scene scene;
    ernie::Camera::Define(scene, "/World/Camera")
        .SetSensorNoise(false);
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
                                     {0.0, 0.0, 1.0}));
    ernie::DistantLight::Define(scene, "/World/Sun")
        .SetShadowEnable(false);
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({0.2, -0.5, 1.0}));
    std::vector<float> positions;
    for (float y = 6.0f; y <= 400.0f; y += 4.0f) {
        positions.insert(positions.end(), {0.0f, y, 0.0f});
    }
    ernie::PointInstancer::Define(scene, "/World/Balls")
        .SetPositions(Tensor::Copy<float>(
            positions,
            {static_cast<std::int64_t>(positions.size() / 3), 3}));
    scene.Set("/World/Balls", tokens::kPrototypes,
              std::vector<Path>{"/World/Balls/Ball"});
    ernie::Sphere::Define(scene, "/World/Balls/Ball").SetRadius(1.0);
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(64, 64));
    return scene;
}

}  // namespace

TEST_F(RasterDelegateTest,
       DistantInstancesDrawCoarserLevelsOfDetail) {
    Scene scene = RecedingSpheres();
    RasterDelegate delegate(*context_);
    const Tensor detailed = [&] {
        ernie::RenderSettings(scene, "/Render/Settings")
            .SetLodMeshError(0.0);
        return delegate.Render(scene);
    }();
    std::vector<std::vector<std::uint32_t>> levels =
        delegate.Meshes().CulledLevelCounts();
    ASSERT_EQ(levels.size(), 1u);
    const std::uint32_t visible = levels[0][0];
    EXPECT_GT(visible, 10u);
    for (std::size_t l = 1; l < levels[0].size(); ++l) {
        EXPECT_EQ(levels[0][l], 0u)
            << "an error of 0 draws the full meshes";
    }

    ernie::RenderSettings(scene, "/Render/Settings")
        .SetLodMeshError(1.0);
    const Tensor simplified = delegate.Render(scene);
    levels = delegate.Meshes().CulledLevelCounts();
    ASSERT_EQ(levels.size(), 1u);
    ASSERT_GT(levels[0].size(), 3u)
        << "the sphere has a chain of levels";
    std::uint32_t total = 0;
    for (const std::uint32_t count : levels[0]) {
        total += count;
    }
    EXPECT_EQ(total, visible)
        << "the same instances, at different levels";
    EXPECT_LT(levels[0][0], visible / 2)
        << "most spheres are far enough to simplify";
    EXPECT_GT(levels[0].back() + levels[0][levels[0].size() - 2], 0u)
        << "the farthest use the coarsest levels";

    // Within a pixel of error the image barely changes.
    const auto* a = detailed.As<std::uint8_t>();
    const auto* b = simplified.As<std::uint8_t>();
    double difference = 0.0;
    for (std::int64_t i = 0; i < detailed.Elements(); ++i) {
        difference +=
            std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    }
    EXPECT_LT(difference / static_cast<double>(detailed.Elements()),
              2.0);
}

TEST(MeshDrawableLevels, SimplifyOnlyMeshesWorthIt) {
    std::unique_ptr<ernie::VulkanContext> context;
    try {
        context = ernie::VulkanContext::Create();
    } catch (const std::runtime_error& error) {
        GTEST_SKIP() << "No Vulkan device: " << error.what();
    }
    ernie::RasterRenderer renderer(*context);
    ernie::MeshDrawable meshes(renderer);
    const std::uint32_t sphere =
        meshes.AddMesh(ernie::SphereMesh(1.0f));
    const std::uint32_t quad =
        meshes.AddMesh(ernie::BoxMesh({2.0f, 2.0f, 2.0f}));
    EXPECT_GT(meshes.LevelCount(sphere), 3u);
    EXPECT_LE(meshes.LevelCount(sphere),
              ernie::MeshDrawable::kMaxLevels);
    EXPECT_EQ(meshes.LevelCount(quad), 1u)
        << "12 triangles: only the full mesh";
}

TEST_F(RasterDelegateTest, GpuCulledInstancersShareOneMultiDraw) {
    Scene scene = CullingScene();  // a quad, and two instancers of
                                   // the same mesh and material
    RasterDelegate delegate(*context_);
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetCulling(tokens::kCpu);
    const Tensor direct = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().DrawCalls(0), 3u)
        << "one call per draw";
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetCulling(tokens::kGpu);
    const Tensor batched = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().DrawCalls(0),
              context_->SupportsMultiDrawIndirect() ? 2u : 3u)
        << "the quad, then both instancers in one multi-draw";
    EXPECT_EQ(Bytes(batched), Bytes(direct));
}

namespace {

/// A camera at the origin looking along +y at a wall 20 m away (a
/// mesh, or with `instancedWall` an instance of a cube), with spheres
/// hidden behind it at 40 m and one in front at 10 m.
Scene WallScene(bool instancedWall) {
    Scene scene;
    ernie::Camera::Define(scene, "/World/Camera")
        .SetSensorNoise(false);
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
                                     {0.0, 0.0, 1.0}));
    ernie::DistantLight::Define(scene, "/World/Sun")
        .SetShadowEnable(false);
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform({0.2, -1.0, 0.5}));
    if (instancedWall) {
        ernie::PointInstancer::Define(scene, "/World/Walls")
            .SetPositions(Tensor::Copy<float>(
                std::vector<float>{0, 20, 0}, {1, 3}));
        scene.Set("/World/Walls", tokens::kPrototypes,
                  std::vector<Path>{"/World/Walls/Wall"});
        ernie::Cube::Define(scene, "/World/Walls/Wall").SetSize(60.0);
        scene.Set(
            "/World/Walls/Wall", tokens::kTransform,
            glm::scale(glm::dmat4(1.0), glm::dvec3(1.0, 0.05, 1.0)));
    } else {
        DefineQuad(scene, "/World/Wall", 30.0f);
        scene.Set("/World/Wall", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0),
                                 glm::dvec3(0.0, 20.0, 0.0)) *
                      glm::rotate(glm::dmat4(1.0), glm::radians(90.0),
                                  glm::dvec3(1.0, 0.0, 0.0)));
    }
    std::vector<float> positions{0, 10, 0};  // in front of the wall
    for (int i = -4; i <= 4; ++i) {
        positions.insert(positions.end(),
                         {static_cast<float>(i), 40.0f, 0.0f});
    }
    ernie::PointInstancer::Define(scene, "/World/Balls")
        .SetPositions(Tensor::Copy<float>(positions, {10, 3}));
    scene.Set("/World/Balls", tokens::kPrototypes,
              std::vector<Path>{"/World/Balls/Ball"});
    ernie::Sphere::Define(scene, "/World/Balls/Ball").SetRadius(0.5);
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(64, 64));
    return scene;
}

}  // namespace

TEST(OcclusionOf, ReadsTheSettingsToken) {
    Scene scene;
    scene.Define("/Render/Settings", tokens::kRenderSettings);
    EXPECT_EQ(ernie::OcclusionOf(scene, "/Render/Settings"),
              ernie::Occlusion::Off);
    scene.Set("/Render/Settings", tokens::kOcclusion,
              std::string(tokens::kOccluders));
    EXPECT_EQ(ernie::OcclusionOf(scene, "/Render/Settings"),
              ernie::Occlusion::Occluders);
    scene.Set("/Render/Settings", tokens::kOcclusion,
              std::string(tokens::kTwoPass));
    EXPECT_EQ(ernie::OcclusionOf(scene, "/Render/Settings"),
              ernie::Occlusion::TwoPass);
    scene.Set("/Render/Settings", tokens::kOcclusion,
              std::string("sometimes"));
    EXPECT_THROW(ernie::OcclusionOf(scene, "/Render/Settings"),
                 std::invalid_argument);
}

TEST_F(RasterDelegateTest, OccludersHideTheInstancesBehindThem) {
    Scene scene = WallScene(false);
    RasterDelegate delegate(*context_);
    const Tensor unculled = delegate.Render(scene);
    EXPECT_EQ(delegate.Meshes().CulledInstanceCounts(),
              std::vector<std::uint32_t>{10u})
        << "frustum culling keeps every ball";

    ernie::RenderSettings(scene, "/Render/Settings")
        .SetOcclusion(tokens::kOccluders);
    const Tensor occluded = delegate.Render(scene);
    EXPECT_TRUE(delegate.Meshes().CullsOcclusion());
    EXPECT_EQ(delegate.Meshes().CulledInstanceCounts(),
              std::vector<std::uint32_t>{1u})
        << "only the ball in front of the wall";
    EXPECT_EQ(Bytes(occluded), Bytes(unculled))
        << "hidden instances change nothing";
}

TEST_F(RasterDelegateTest,
       OccludersHideInstancesFromAPushbroomsStrip) {
    // The wall scene scanned by a pushbroom rising past it: instances
    // test against its strip's depth pyramid through the strip's
    // projection.
    Scene scene = WallScene(false);
    ernie::Camera(scene, "/World/Camera").SetScan(tokens::kPushbroom);
    const auto scan = [&](RasterDelegate& delegate) {
        const auto at = [](double z) {
            return ernie::LookAtTransform(
                {0.0, 0.0, z}, {0.0, 1.0, z}, {0.0, 0.0, 1.0});
        };
        scene.Set("/World/Camera", tokens::kTransform, at(-3.0));
        delegate.Render(scene);  // the strip's start
        scene.Set("/World/Camera", tokens::kTransform, at(3.0));
        return delegate.Render(scene);
    };
    RasterDelegate delegate(*context_);
    const Tensor unculled = scan(delegate);
    EXPECT_EQ(delegate.Meshes().CulledInstanceCounts(),
              std::vector<std::uint32_t>{10u});

    ernie::RenderSettings(scene, "/Render/Settings")
        .SetOcclusion(tokens::kOccluders);
    const Tensor occluded = scan(delegate);
    EXPECT_TRUE(delegate.Meshes().CullsOcclusion());
    EXPECT_EQ(delegate.Meshes().CulledInstanceCounts(),
              std::vector<std::uint32_t>{1u})
        << "only the ball in front of the wall";
    EXPECT_EQ(Bytes(occluded), Bytes(unculled))
        << "hidden instances change nothing";

    // With the wall gone the strip sees them all again.
    scene.Remove("/World/Wall");
    scan(delegate);
    EXPECT_EQ(delegate.Meshes().CulledInstanceCounts(),
              std::vector<std::uint32_t>{10u});
}

TEST_F(RasterDelegateTest,
       TwoPassOcclusionLetsInstancesHideEachOther) {
    Scene scene = WallScene(true);
    RasterDelegate delegate(*context_);
    const Tensor unculled = delegate.Render(scene);
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetOcclusion(tokens::kOccluders);
    delegate.Render(scene);
    // Jobs in draw order: the balls, then the wall.
    EXPECT_EQ(delegate.Meshes().CulledInstanceCounts(),
              (std::vector<std::uint32_t>{10u, 1u}))
        << "an instanced wall occludes nothing in a single pass";

    ernie::RenderSettings(scene, "/Render/Settings")
        .SetOcclusion(tokens::kTwoPass);
    for (int frame = 0; frame < 3; ++frame) {
        const Tensor image = delegate.Render(scene);
        EXPECT_EQ(Bytes(image), Bytes(unculled)) << "frame " << frame;
    }
    // By now the wall, visible last frame, draws first and hides the
    // balls behind it.
    EXPECT_EQ(delegate.Meshes().CulledInstanceCounts(),
              (std::vector<std::uint32_t>{1u, 1u}));
}

TEST_F(RasterDelegateTest, OcclusionCullsEveryChunkOfABatch) {
    Scene scene = WallScene(false);
    const std::uint32_t views =
        ernie::RenderTargets::MaxViews(*context_, 16) + 3;
    std::vector<double> poses;
    for (std::uint32_t v = 0; v < views; ++v) {
        // Views sidestep along x, all still facing the wall.
        const glm::dmat4 pose = ernie::LookAtTransform(
            {0.01 * v, 0.0, 0.0}, {0.01 * v, 1.0, 0.0},
            {0.0, 0.0, 1.0});
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poses.push_back(pose[col][row]);
            }
        }
    }
    ernie::Camera(scene, "/World/Camera")
        .SetPoses(Tensor::Copy<double>(
            poses, {static_cast<std::int64_t>(views), 4, 4}));
    ernie::RenderProduct::Define(scene, "/Render/Batch")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(16, 16));
    RasterDelegate delegate(*context_);
    const Path batchOnly[] = {"/Render/Batch"};
    const auto render = [&] {
        return delegate.Step(scene, batchOnly)
            .at("/Render/Batch")
            .vars.at("LdrColor")
            .Map(ernie::MapDevice::Cpu)
            .tensor;
    };
    const Tensor unculled = render();
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetOcclusion(tokens::kOccluders);
    const Tensor occluded = render();
    EXPECT_EQ(Bytes(occluded), Bytes(unculled));
    const std::vector<std::uint32_t> kept =
        delegate.Meshes().CulledInstanceCounts();
    ASSERT_EQ(kept.size(), views) << "one job per view";
    EXPECT_EQ(kept.front(), 1u);
    EXPECT_EQ(kept.back(), 1u)
        << "the last chunk's views cull against their own pyramids";
}

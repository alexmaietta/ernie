// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/cube_sphere_lod.h"
#include "ernie/raster/delegate.h"
#include "ernie/raster/planet_drawable.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <vector>

using ernie::CubeSphereLod;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

constexpr double kMoon = 1737400.0;

/// Heights of a planet with faces of `n` quads: rolling relief of a
/// few kilometres sampled by direction, so the faces agree where they
/// meet.
std::vector<float> PlanetHeights(std::uint32_t n) {
    const CubeSphereLod shape(kMoon, n, 4);
    std::vector<float> heights;
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces;
         ++face) {
        for (std::uint32_t row = 0; row <= n; ++row) {
            for (std::uint32_t col = 0; col <= n; ++col) {
                const glm::dvec3 d =
                    shape.VertexDirection(face, col, row);
                heights.push_back(
                    static_cast<float>(3000.0 * std::sin(9.0 * d.x) *
                                       std::cos(7.0 * d.y + d.z)));
            }
        }
    }
    return heights;
}

Scene PlanetScene(std::uint32_t n) {
    const std::vector<float> heights = PlanetHeights(n);
    const auto side = static_cast<std::int64_t>(n + 1);
    Scene scene;
    ernie::Terrain::Define(scene, "/World/Moon")
        .SetHeights(Tensor::Copy<float>(heights, {6, side, side}))
        .SetPlanetRadius(kMoon)
        .SetMaterialBinding("/Looks/Regolith");
    ernie::Material::Define(scene, "/Looks/Regolith")
        .SetAlbedo(glm::dvec3(0.5));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform,
             ernie::DistantLightTransform({1.0, 0.3, 0.2}));
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kSize, kSize))
        .SetClearColor(glm::dvec3(0.0, 0.0, 1.0));
    return scene;
}

void PlaceCamera(Scene& scene, const glm::dvec3& eye,
                 const glm::dvec3& target, const glm::dvec3& up,
                 glm::dvec2 clipping) {
    ernie::Camera::Define(scene, "/World/Camera")
        .SetFocalLength(18.0)
        .SetClippingRange(clipping);
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform(eye, target, up));
}

bool IsClear(const std::uint8_t* texel) {
    return texel[0] == 0 && texel[1] == 0 && texel[2] == 255;
}

std::size_t Covered(const Tensor& image) {
    const auto* pixels = image.As<std::uint8_t>();
    std::size_t covered = 0;
    for (std::int64_t i = 0; i < image.Elements(); i += 4) {
        covered += IsClear(pixels + i) ? 0u : 1u;
    }
    return covered;
}

}  // namespace

TEST_F(RasterDelegateTest, APlanetFromOrbitIsADiscOfItsAngularSize) {
    Scene scene = PlanetScene(128);
    const double distance = 4.0 * kMoon;
    PlaceCamera(scene, {distance, 0.0, 0.0}, {0.0, 0.0, 0.0},
                {0.0, 0.0, 1.0}, {1e4, 1e8});
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    ASSERT_TRUE(delegate.Planet().HasHeights());
    EXPECT_FALSE(delegate.Terrain().HasHeightfield());

    const Tensor image = delegate.Render(scene);
    const auto* pixels = image.As<std::uint8_t>();
    const std::size_t centre = (kSize / 2 * kSize + kSize / 2) * 4;
    EXPECT_FALSE(IsClear(pixels + centre));
    EXPECT_TRUE(IsClear(pixels));
    EXPECT_GT(delegate.Planet().PatchCount(), 0u);

    // An 18 mm lens on the default 15.2908 mm vertical aperture: the
    // disc's radius in pixels.
    const double halfWidth = 0.5 * 15.2908 / 18.0;
    const double discRadius = std::tan(std::asin(kMoon / distance)) /
                              halfWidth * 0.5 *
                              static_cast<double>(kSize);
    const double expected =
        std::numbers::pi * discRadius * discRadius;
    EXPECT_NEAR(static_cast<double>(Covered(image)), expected,
                0.12 * expected);
}

// An oblate planet from over its equator: the silhouette's height
// over its width is c / a, the tangents from the eye to an ellipse of
// semi-axes a and c leaving at angles whose tangents are a and c over
// sqrt(D^2 - a^2).
TEST_F(RasterDelegateTest, AnEllipsoidPlanetFromOrbitIsItsShape) {
    Scene scene = PlanetScene(128);
    ernie::Terrain(scene, "/World/Moon")
        .SetPlanetRadii(glm::dvec3(kMoon, kMoon, 0.7 * kMoon));
    PlaceCamera(scene, {3.0 * kMoon, 0.0, 0.0}, {0.0, 0.0, 0.0},
                {0.0, 0.0, 1.0}, {1e4, 1e8});
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    const Tensor image = delegate.Render(scene);
    const auto* pixels = image.As<std::uint8_t>();
    std::size_t wide = 0;
    std::size_t tall = 0;
    for (std::uint32_t i = 0; i < kSize; ++i) {
        wide +=
            IsClear(pixels + (kSize / 2 * kSize + i) * 4) ? 0u : 1u;
        tall +=
            IsClear(pixels + (i * kSize + kSize / 2) * 4) ? 0u : 1u;
    }
    ASSERT_GT(wide, kSize / 8);
    EXPECT_NEAR(static_cast<double>(tall) / static_cast<double>(wide),
                0.7, 0.05);
}

TEST_F(RasterDelegateTest,
       APlanetSeenFromLowAltitudeFillsBelowTheHorizon) {
    Scene scene = PlanetScene(256);
    const glm::dvec3 up = glm::normalize(glm::dvec3(0.2, 0.9, 0.4));
    const glm::dvec3 east =
        glm::normalize(glm::cross(glm::dvec3(0.0, 0.0, 1.0), up));
    const glm::dvec3 eye = up * (kMoon + 8000.0);
    PlaceCamera(scene, eye, eye + east, up, {1.0, 1e7});
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    const Tensor image = delegate.Render(scene);
    const auto* pixels = image.As<std::uint8_t>();
    std::size_t topClear = 0;
    std::size_t bottomGround = 0;
    for (std::uint32_t col = 0; col < kSize; ++col) {
        topClear += IsClear(pixels + col * 4) ? 1u : 0u;
        bottomGround +=
            IsClear(pixels + ((kSize - 1) * kSize + col) * 4) ? 0u
                                                              : 1u;
    }
    EXPECT_EQ(topClear, kSize) << "sky above";
    EXPECT_EQ(bottomGround, kSize) << "ground below";
    EXPECT_LT(delegate.Planet().PatchCount(), 4000u)
        << "the far side is culled";
}

// An Earth of 156 km quads from 2 m up: refined below its data, the
// ground is whole to the horizon, without facets sagging under the
// eye or cracks along the face's centre lines.
TEST_F(RasterDelegateTest, APlanetRefinesBelowItsDataUnderTheEye) {
    constexpr double kEarth = 6371008.8;
    Scene scene;
    ernie::Terrain::Define(scene, "/World/Earth")
        .SetHeights(Tensor::Copy<float>(
            std::vector<float>(6 * 65 * 65, 0.0f), {6, 65, 65}))
        .SetPlanetRadius(kEarth)
        .SetMaterialBinding("/Looks/Ground");
    ernie::Material::Define(scene, "/Looks/Ground")
        .SetAlbedo(glm::dvec3(0.5));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform,
             ernie::DistantLightTransform({0.5, 0.0, 0.866}));
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kSize, kSize))
        .SetClearColor(glm::dvec3(0.0, 0.0, 1.0));
    // Over face 4's centre, looking along its centre line.
    const glm::dvec3 eye(0.0, 0.0, kEarth + 2.0);
    PlaceCamera(scene, eye, eye + glm::dvec3(-1000.0, 0.0, -400.0),
                {0.0, 0.0, 1.0}, {0.1, 1e8});
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    const Tensor image = delegate.Render(scene);
    const auto* pixels = image.As<std::uint8_t>();
    std::size_t clear = 0;
    for (std::uint32_t row = kSize / 2; row < kSize; ++row) {
        for (std::uint32_t col = 0; col < kSize; ++col) {
            clear +=
                IsClear(pixels + (row * kSize + col) * 4) ? 1u : 0u;
        }
    }
    EXPECT_EQ(clear, 0u) << "holes in the ground below the horizon";
    EXPECT_GT(delegate.Planet().Lod().Refinement(), 0u);
    // Shadows fit to what can be seen of it from 2 m, past the
    // horizon 5 km away and the origin's reach, not the planet.
    const auto casting = delegate.Planet().Bounds();
    const auto body = delegate.Planet().BodyBounds();
    ASSERT_TRUE(casting && body);
    EXPECT_LT(glm::length(casting->second - casting->first), 30e3f);
    EXPECT_GT(glm::length(body->second - body->first), 2e7f);
}

// Sun shadows on an Earth-sized planet, cascades fitted to what a
// frame saw: the cascades' depth spans only their slices, so the
// sunlit ground does not shadow itself in speckles (acne).
TEST_F(RasterDelegateTest, APlanetsSunlitGroundHasNoShadowAcne) {
    constexpr double kEarth = 6371008.8;
    Scene scene;
    ernie::Terrain::Define(scene, "/World/Earth")
        .SetHeights(Tensor::Copy<float>(
            std::vector<float>(6 * 65 * 65, 0.0f), {6, 65, 65}))
        .SetPlanetRadius(kEarth)
        .SetMaterialBinding("/Looks/Ground");
    scene.Set("/World/Earth", tokens::kTransform,
              glm::translate(glm::dmat4(1.0),
                             glm::dvec3(0.0, 0.0, -kEarth)));
    ernie::Material::Define(scene, "/Looks/Ground")
        .SetAlbedo(glm::dvec3(0.5));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform,
             ernie::DistantLightTransform({0.5, 0.0, 0.866}));
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kSize, kSize));
    PlaceCamera(scene, {-1.7, 5.2, 14.7}, {-25.0, 0.0, 1.0},
                {0.0, 0.0, 1.0}, {1.0, 1e6});
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    for (int frame = 0; frame < 3; ++frame) {
        const Tensor image = delegate.Render(scene);
        const auto* pixels = image.As<std::uint8_t>();
        int lowest = 255;
        int highest = 0;
        for (std::uint32_t i = 0; i < kSize * kSize; ++i) {
            lowest = std::min<int>(lowest, pixels[i * 4 + 1]);
            highest = std::max<int>(highest, pixels[i * 4 + 1]);
        }
        EXPECT_LE(highest - lowest, 8) << "frame " << frame;
    }
}

TEST_F(RasterDelegateTest,
       APlanetStaysStillUnderCentimetreCameraMoves) {
    Scene scene = PlanetScene(256);
    const glm::dvec3 up = glm::normalize(glm::dvec3(-0.4, 0.3, 0.8));
    const glm::dvec3 north =
        glm::normalize(glm::dvec3(0.0, 0.0, 1.0) - up * up.z);
    const glm::dvec3 eye = up * (kMoon + 3000.0 + 400.0);
    const glm::dvec3 target = eye + 2000.0 * north - 600.0 * up;
    PlaceCamera(scene, eye, target, up, {1.0, 1e7});
    RasterDelegate delegate(*context_);
    delegate.Sync(scene);
    const std::vector<std::uint8_t> before =
        Bytes(delegate.Render(scene));
    PlaceCamera(scene, eye + glm::dvec3(0.01, 0.0, 0.0),
                target + glm::dvec3(0.01, 0.0, 0.0), up, {1.0, 1e7});
    const std::vector<std::uint8_t> after =
        Bytes(delegate.Render(scene));
    EXPECT_GT(Covered(Tensor::Copy<std::uint8_t>(before,
                                                 {kSize, kSize, 4})),
              kSize * kSize / 4);
    EXPECT_LE(BytesDiffering(before, after), before.size() / 100);
}

TEST_F(RasterDelegateTest,
       PlanetsNeedFacesOfAtLeastALeafAndSixLayers) {
    ernie::RasterRenderer renderer(*context_,
                                   VK_FORMAT_R8G8B8A8_UNORM);
    ernie::PlanetDrawable planet(renderer);
    std::vector<float> small(33 * 33, 0.0f);
    std::array<ernie::RasterView, 6> faces;
    faces.fill(ernie::RasterView{small.data(), 33, 33});
    EXPECT_THROW(planet.SetHeights(kMoon, faces),
                 std::invalid_argument);
    EXPECT_FALSE(planet.HasHeights());
    EXPECT_FALSE(planet.BodyBounds().has_value());

    std::vector<float> enough(65 * 65, 1.0f);
    faces.fill(ernie::RasterView{enough.data(), 65, 65});
    planet.SetHeights(kMoon, faces);
    ASSERT_TRUE(planet.HasHeights());
    EXPECT_EQ(planet.Lod().Levels(), 1u);
    const auto bounds = planet.BodyBounds();
    ASSERT_TRUE(bounds.has_value());
    EXPECT_NEAR(bounds->second.x, kMoon + 1.0, 1.0);
    EXPECT_GT(planet.GpuBytes(), 6u * 65u * 65u * 4u);
    planet.ClearHeights();
    EXPECT_FALSE(planet.HasHeights());

    Scene scene = PlanetScene(64);
    ernie::Terrain(scene, "/World/Moon")
        .SetHeights(Tensor::Copy<float>(enough, {1, 65, 65}));
    RasterDelegate delegate(*context_);
    EXPECT_THROW(delegate.Sync(scene), std::exception);
}

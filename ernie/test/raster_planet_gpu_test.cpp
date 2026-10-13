// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/cube_sphere_lod.h"
#include "ernie/geometry/planet_heights.h"
#include "ernie/raster/delegate.h"
#include "ernie/raster/planet_drawable.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/dem.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/raster_source.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <thread>
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

// A planet's albedo texture wraps the body in latitude and longitude:
// a map coloured by quarter of longitude (blue, red, green, white
// from 180 degrees west) shows red over 45 degrees west and green
// over 45 degrees east.
TEST_F(RasterDelegateTest, APlanetsAlbedoTextureWrapsItByLatLong) {
    constexpr std::int64_t kWidth = 64;
    constexpr std::int64_t kHeight = 32;
    const std::array<std::array<std::uint8_t, 3>, 4> quarters{
        {{0, 0, 255}, {255, 0, 0}, {0, 255, 0}, {255, 255, 255}}};
    std::vector<std::uint8_t> texels;
    for (std::int64_t row = 0; row < kHeight; ++row) {
        for (std::int64_t col = 0; col < kWidth; ++col) {
            const auto& c =
                quarters[static_cast<std::size_t>(col * 4 / kWidth)];
            texels.insert(texels.end(), {c[0], c[1], c[2], 255});
        }
    }
    Scene scene = PlanetScene(64);
    ernie::Shader::Define(scene, "/Looks/Regolith/Surface")
        .SetInfoId(tokens::kUsdPreviewSurface)
        .SetDiffuseColor(glm::dvec3(0.5));
    ernie::Shader::Define(scene, "/Looks/Regolith/Tex")
        .SetInfoId(tokens::kUsdUVTexture)
        .SetSourceColorSpace("raw")
        .SetTexture(
            Tensor::Copy<std::uint8_t>(texels, {kHeight, kWidth, 4}));
    ernie::Connect(scene, "/Looks/Regolith", tokens::kSurfaceOutput,
                   "/Looks/Regolith/Surface", tokens::kSurfaceOutput);
    ernie::Connect(scene, "/Looks/Regolith/Surface",
                   tokens::kDiffuseColor, "/Looks/Regolith/Tex",
                   "outputs:rgb");

    RasterDelegate delegate(*context_);
    const auto centre = [&](double longitude) {
        const glm::dvec3 towards(std::cos(longitude),
                                 std::sin(longitude), 0.0);
        scene.Define("/World/Sun", tokens::kDistantLight)
            .Set(tokens::kTransform,
                 ernie::DistantLightTransform(towards));
        PlaceCamera(scene, 4.0 * kMoon * towards, {0.0, 0.0, 0.0},
                    {0.0, 0.0, 1.0}, {1e4, 1e8});
        const Tensor image = delegate.Render(scene);
        const auto* texel = image.As<std::uint8_t>() +
                            (kSize / 2 * kSize + kSize / 2) * 4;
        return std::array<int, 3>{texel[0], texel[1], texel[2]};
    };
    const double quarter = std::numbers::pi / 4.0;
    const auto west = centre(-quarter);
    EXPECT_GT(west[0], 2 * std::max(west[1], west[2]) + 10);
    const auto east = centre(quarter);
    EXPECT_GT(east[1], 2 * std::max(east[0], east[2]) + 10);
}

// A Moon from an 8192 x 4096 height file (LOLA's LDEM layout: int16
// half metres about the radius, from 0 degrees east): faces of 1024
// quads read at once and pages of the file's 2048 paged in. From 20
// km up the ground draws as a planet of the file's 2048-quad faces
// does, waiting for every page; frames that do not wait draw coarser
// ones first and come to the same.
TEST_F(RasterDelegateTest, APlanetFromAHeightsFilePagesInItsDetail) {
    constexpr std::uint32_t kWidth = 8192;
    constexpr std::uint32_t kHeight = 4096;
    const std::filesystem::path data =
        std::filesystem::path(::testing::TempDir()) /
        "paged_ldem.img";
    {
        std::vector<std::int16_t> samples(std::size_t{kWidth} *
                                          kHeight);
        for (std::uint32_t row = 0; row < kHeight; ++row) {
            const double latitude =
                (90.0 - (row + 0.5) * 180.0 / kHeight) *
                std::numbers::pi / 180.0;
            for (std::uint32_t col = 0; col < kWidth; ++col) {
                const double longitude = (col + 0.5) * 360.0 /
                                         kWidth * std::numbers::pi /
                                         180.0;
                const double metres =
                    2000.0 * std::sin(3.0 * latitude) *
                        std::cos(2.0 * longitude) +
                    300.0 * std::sin(2000.0 * latitude) *
                        std::sin(2000.0 * longitude);
                samples[std::size_t{row} * kWidth + col] =
                    static_cast<std::int16_t>(
                        std::lround(2.0 * metres));
            }
        }
        std::ofstream(data, std::ios::binary)
            .write(reinterpret_cast<const char*>(samples.data()),
                   static_cast<std::streamsize>(samples.size() * 2));
        std::ofstream(data.parent_path() / "paged_ldem.lbl")
            << "PDS_VERSION_ID = PDS3\n^IMAGE = \"paged_ldem.img\"\n"
               "OBJECT = IMAGE\n  LINES = 4096\n  LINE_SAMPLES = "
               "8192\n"
               "  SAMPLE_TYPE = LSB_INTEGER\n  SAMPLE_BITS = 16\n"
               "  SCALING_FACTOR = 0.5\n  OFFSET = 1737400\n"
               "END_OBJECT = IMAGE\nOBJECT = IMAGE_MAP_PROJECTION\n"
               "  MAP_PROJECTION_TYPE = \"SIMPLE CYLINDRICAL\"\n"
               "  MAP_RESOLUTION = 22.7555555555555556\n"
               "  CENTER_LATITUDE = 0\n  MAXIMUM_LATITUDE = 90\n"
               "  WESTERNMOST_LONGITUDE = 0\n"
               "END_OBJECT = IMAGE_MAP_PROJECTION\nEND\n";
    }
    const auto scene = [&](bool file, std::string_view loading) {
        Scene s = PlanetScene(64);
        ernie::Terrain moon(s, "/World/Moon");
        if (file) {
            s.Clear("/World/Moon", tokens::kHeights);
            moon.SetHeightsFile(data.string())
                .SetHeightsReference(kMoon);
        } else {
            ernie::PlanetHeightsOptions options;
            options.reference = kMoon;
            const ernie::PlanetTile faces = ernie::PlanetFacesOf(
                *ernie::OpenRaster(data), options, glm::dvec3(kMoon),
                2048);
            moon.SetHeights(
                Tensor::Copy<float>(faces.heights, {6, 2049, 2049}));
        }
        ernie::RenderSettings(s, "/Render/Settings")
            .SetTextureLoading(std::string(loading));
        // Shaded by its slopes alone, against a low light.
        moon.SetColorMode(std::string(tokens::kHillshade))
            .SetHillshadeAltitude(20.0);
        // 20 km over 30 N 40 E, looking down the slope to the east.
        const glm::dvec3 up = ernie::BodyDirection(30.0, 40.0);
        const glm::dvec3 east =
            glm::normalize(glm::cross(glm::dvec3(0.0, 0.0, 1.0), up));
        const glm::dvec3 eye = up * (kMoon + 20000.0);
        PlaceCamera(s, eye, eye + east * 30000.0 - up * 20000.0, up,
                    {10.0, 1e7});
        return s;
    };

    Scene tensor = scene(false, tokens::kComplete);
    RasterDelegate reference(*context_);
    const Tensor expected = reference.Render(tensor);
    ASSERT_GT(Covered(expected), std::size_t{kSize} * kSize * 9 / 10)
        << "ground fills the view";

    Scene file = scene(true, tokens::kComplete);
    RasterDelegate complete(*context_);
    const Tensor waited = complete.Render(file);
    const ernie::PlanetPages* pages = complete.Planet().Pages();
    ASSERT_NE(pages, nullptr);
    EXPECT_EQ(pages->Depth(), 1u);
    EXPECT_GT(pages->Resident(), 0u);
    EXPECT_EQ(pages->Pending(), 0u);
    // Mean difference of the images' channels, in 8-bit steps.
    const auto differ = [](const Tensor& a, const Tensor& b) {
        double sum = 0.0;
        for (std::int64_t i = 0; i < a.Elements(); ++i) {
            sum += std::abs(int(a.As<std::uint8_t>()[i]) -
                            int(b.As<std::uint8_t>()[i]));
        }
        return sum / static_cast<double>(a.Elements());
    };
    const double paged = differ(waited, expected);
    EXPECT_LT(paged, 2.0);
    // The faces alone, without their pages, miss the file's detail.
    {
        Scene coarse = scene(false, tokens::kComplete);
        ernie::PlanetHeightsOptions options;
        options.reference = kMoon;
        const ernie::PlanetTile faces =
            ernie::PlanetFacesOf(*ernie::OpenRaster(data), options,
                                 glm::dvec3(kMoon), 1024);
        ernie::Terrain(coarse, "/World/Moon")
            .SetHeights(
                Tensor::Copy<float>(faces.heights, {6, 1025, 1025}));
        RasterDelegate delegate(*context_);
        const double alone =
            differ(delegate.Render(coarse), expected);
        EXPECT_GT(alone, 1.0);
        EXPECT_GT(alone, 5.0 * paged);
    }

    Scene streamed = scene(true, tokens::kAsync);
    RasterDelegate async(*context_);
    // The file is read off the frame: frames before it is in draw no
    // planet, then the first after it draws its faces as pages come.
    async.Render(streamed);
    EXPECT_FALSE(async.Planet().HasHeights())
        << "the first frame does not wait for the file";
    for (int frame = 0; frame < 1000 && !async.Planet().HasHeights();
         ++frame) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        async.Render(streamed);
    }
    ASSERT_TRUE(async.Planet().HasHeights());
    ASSERT_NE(async.Planet().Pages(), nullptr);
    EXPECT_GT(async.Planet().Pages()->Pending(), 0u)
        << "its first frame draws the faces";
    Tensor last;
    for (int frame = 0;
         frame < 200 && async.Planet().Pages()->Pending() > 0;
         ++frame) {
        last = async.Render(streamed);
    }
    EXPECT_EQ(async.Planet().Pages()->Pending(), 0u);
    last = async.Render(streamed);
    EXPECT_LT(differ(last, waited), 0.5);
}

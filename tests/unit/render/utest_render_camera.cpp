// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <functional>
#include <numbers>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/scene.h"
#include "support/devices.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

constexpr uint32_t kSize = 8;
constexpr float kWallDistance = 5;
const std::vector<std::string> kProducts = {"/Render/View"};

Ref<Scene> FacingWall(Device& device, const Camera::Desc& camera) {
    const Ref<Scene> scene = Scene::Create(device);
    scene->Define<Plane>("/Wall", {.translate = {0, 0, -kWallDistance}, .width = 0, .length = 0});
    scene->Define<Camera>("/Cam", camera);
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Depth", {.sourceName = "DistanceToCamera"});
    scene->Define<RenderVar>("/Render/Plane", {.sourceName = "DistanceToImagePlane"});
    scene->Define<RenderProduct>("/Render/View",
                                 {.camera = {"/Cam"},
                                  .resolution = {kSize, kSize},
                                  .orderedVars = {"/Render/Depth", "/Render/Plane"}});
    return scene;
}

std::vector<float> Distances(Renderer& renderer, const char* source = "DistanceToCamera") {
    RenderOutput output = renderer.Step(kProducts, 0);
    output.Wait();
    return output["/Render/View"].frames.at(0).Source(source).tensor.ToHost<float>();
}

std::vector<float> WallDistances(Device& device, Float2 clippingRange) {
    Renderer renderer(FacingWall(device, {.focalLength = 10, .clippingRange = clippingRange}));
    return Distances(renderer);
}

TEST(CameraTest, FarClipsByDistanceAlongTheRay) {
    ForEachDevice([](Device& device) {
        const std::vector<float> depth = WallDistances(device, {1, 6});
        ASSERT_EQ(depth.size(), kSize * kSize);
        EXPECT_NEAR(depth[kSize / 2 * kSize + kSize / 2], kWallDistance, 0.1);
        EXPECT_TRUE(std::isinf(depth[0]));
    });
}

TEST(CameraTest, NearClipsByDistanceAlongTheRay) {
    ForEachDevice([](Device& device) {
        const std::vector<float> depth = WallDistances(device, {6, 100});
        ASSERT_EQ(depth.size(), kSize * kSize);
        EXPECT_TRUE(std::isinf(depth[kSize / 2 * kSize + kSize / 2]));
        EXPECT_GT(depth[0], 6);
    });
}

TEST(CameraTest, PerspectiveImagePlaneDistanceIsDepthAlongTheAxis) {
    ForEachDevice([](Device& device) {
        Renderer renderer(FacingWall(device, {.focalLength = 10}));
        const std::vector<float> depth = Distances(renderer, "DistanceToImagePlane");
        ASSERT_EQ(depth.size(), kSize * kSize);
        EXPECT_NEAR(depth[0], kWallDistance, 0.01);
        EXPECT_GT(Distances(renderer)[0], kWallDistance + 1);
    });
}

TEST(CameraTest, OrthographicRaysRunParallelToTheAxis) {
    ForEachDevice([](Device& device) {
        Renderer renderer(FacingWall(device, {.projection = Projection::Orthographic}));
        const std::vector<float> depth = Distances(renderer);
        ASSERT_EQ(depth.size(), kSize * kSize);
        EXPECT_NEAR(depth[0], kWallDistance, 0.01);
        EXPECT_NEAR(depth[kSize * kSize - 1], kWallDistance, 0.01);
        EXPECT_NEAR(Distances(renderer, "DistanceToImagePlane")[0], kWallDistance, 0.01);
    });
}

TEST(CameraTest, DepthOfFieldRaysMeetAtTheFocusDistance) {
    ForEachDevice([](Device& device) {
        Renderer renderer(FacingWall(device, {.fStop = 1, .focusDistance = kWallDistance}));
        const std::vector<float> depth = Distances(renderer, "DistanceToImagePlane");
        ASSERT_EQ(depth.size(), kSize * kSize);
        EXPECT_NEAR(depth[kSize / 2 * kSize + kSize / 2], kWallDistance, 0.01);
    });
}

TEST(CameraTest, MovingTheCameraUpdatesItsRays) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = FacingWall(device, {});
        Renderer renderer(scene);
        const uint32_t center = kSize / 2 * kSize + kSize / 2;
        EXPECT_NEAR(Distances(renderer)[center], kWallDistance, 0.05);
        scene->Select("/Cam").Get<Double3>("translate").Set(Double3{0, 0, 1});
        EXPECT_NEAR(Distances(renderer)[center], kWallDistance + 1, 0.05);
    });
}

constexpr float kFocalLength = 10;
constexpr float kHalfWidth = 20.955f / 2;
constexpr float kHalfHeight = 15.2908f / 2;

Float2 PixelFilm(uint32_t x, uint32_t y, Float2 half = {kHalfWidth, kHalfWidth}) {
    const float ndcX = (float(x) + 0.5f) / kSize * 2 - 1;
    const float ndcY = (float(y) + 0.5f) / kSize * 2 - 1;
    return {ndcX * half.x, -ndcY * half.y};
}

float PixelDepth(const std::vector<float>& depth, uint32_t x, uint32_t y) {
    return depth[y * kSize + x];
}

std::vector<float> DistortedDistances(Device& device, DistortionModel model,
                                      const std::function<void(Prim&)>& configure) {
    const Ref<Scene> scene = FacingWall(device, {.focalLength = kFocalLength});
    Prim camera = *scene->Find("/Cam");
    camera.Set(LensDistortionAPI::kDistortionModel, model);
    configure(camera);
    Renderer renderer(scene);
    return Distances(renderer);
}

float AngularWallDistance(float angle) {
    return kWallDistance / std::cos(angle);
}

TEST(CameraTest, ProductsConformTheFilmToTheirAspectRatio) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = FacingWall(device, {.focalLength = kFocalLength});
        Prim product = *scene->Find("/Render/View");
        Renderer renderer(scene);
        const auto corner = [&](AspectRatioConformPolicy policy, float pixelAspect) {
            product.Set(RenderProduct::kAspectRatioConformPolicy, policy);
            product.Set(RenderProduct::kPixelAspectRatio, pixelAspect);
            return PixelDepth(Distances(renderer), 0, 0);
        };
        const auto expected = [](Float2 half) {
            const Float2 film = PixelFilm(0, 0, half);
            return kWallDistance * std::hypot(1.0f, film.x / kFocalLength, film.y / kFocalLength);
        };
        EXPECT_NEAR(corner(AspectRatioConformPolicy::ExpandAperture, 1),
                    expected({kHalfWidth, kHalfWidth}), 0.01);
        EXPECT_NEAR(corner(AspectRatioConformPolicy::CropAperture, 1),
                    expected({kHalfHeight, kHalfHeight}), 0.01);
        EXPECT_NEAR(corner(AspectRatioConformPolicy::AdjustApertureWidth, 1),
                    expected({kHalfHeight, kHalfHeight}), 0.01);
        EXPECT_NEAR(corner(AspectRatioConformPolicy::AdjustApertureHeight, 1),
                    expected({kHalfWidth, kHalfWidth}), 0.01);
        EXPECT_NEAR(corner(AspectRatioConformPolicy::AdjustPixelAspectRatio, 1),
                    expected({kHalfWidth, kHalfHeight}), 0.01);
        EXPECT_NEAR(corner(AspectRatioConformPolicy::ExpandAperture, 2),
                    expected({2 * kHalfHeight, kHalfHeight}), 0.01);
    });
}

TEST(CameraTest, KannalaBrandtWithoutCoefficientsIsEquidistant) {
    ForEachDevice([](Device& device) {
        const std::vector<float> depth =
            DistortedDistances(device, DistortionModel::KannalaBrandt, [](Prim&) {});
        const Float2 film = PixelFilm(7, 4);
        const float angle = std::hypot(film.x, film.y) / kFocalLength;
        EXPECT_NEAR(PixelDepth(depth, 7, 4), AngularWallDistance(angle), 0.02);
    });
}

TEST(CameraTest, KannalaBrandtBendsTheAngleByItsPolynomial) {
    ForEachDevice([](Device& device) {
        const float k1 = 0.1f;
        const std::vector<float> depth =
            DistortedDistances(device, DistortionModel::KannalaBrandt, [&](Prim& camera) {
                camera.Set(LensDistortionAPI::kDistortionRadial, Float4{k1, 0, 0, 0});
            });
        const Float2 film = PixelFilm(7, 4);
        const float radius = std::hypot(film.x, film.y) / kFocalLength;
        float angle = radius;
        for (int step = 0; step < 20; ++step) {
            angle -= (angle * (1 + k1 * angle * angle) - radius) / (1 + 3 * k1 * angle * angle);
        }
        EXPECT_NEAR(PixelDepth(depth, 7, 4), AngularWallDistance(angle), 0.02);
    });
}

TEST(CameraTest, FisheyeSeesNothingPastItsMaxFieldOfView) {
    ForEachDevice([](Device& device) {
        const std::vector<float> depth =
            DistortedDistances(device, DistortionModel::KannalaBrandt, [](Prim& camera) {
                camera.Set(LensDistortionAPI::kDistortionMaxFieldOfView, 90.0f);
            });
        EXPECT_TRUE(std::isinf(PixelDepth(depth, 7, 4)));
        EXPECT_FALSE(std::isinf(PixelDepth(depth, 4, 4)));
    });
}

TEST(CameraTest, FThetaMapsTheAngleToRadiusByItsPolynomial) {
    ForEachDevice([](Device& device) {
        const std::vector<float> terms = {0, 8, 1.5f};
        const std::vector<float> depth =
            DistortedDistances(device, DistortionModel::FTheta, [&](Prim& camera) {
                camera.Set(LensDistortionAPI::kDistortionFTheta, terms);
            });
        const Float2 film = PixelFilm(6, 1);
        const float radius = std::hypot(film.x, film.y);
        const float angle =
            (-terms[1] + std::sqrt(terms[1] * terms[1] + 4 * terms[2] * radius)) / (2 * terms[2]);
        EXPECT_NEAR(PixelDepth(depth, 6, 1), AngularWallDistance(angle), 0.02);
    });
}

TEST(CameraTest, BrownConradyUndistortsTheFilmBeforeProjecting) {
    ForEachDevice([](Device& device) {
        const float k1 = -0.05f;
        const std::vector<float> depth =
            DistortedDistances(device, DistortionModel::BrownConrady, [&](Prim& camera) {
                camera.Set(LensDistortionAPI::kDistortionRadial, Float4{k1, 0, 0, 0});
            });
        const Float2 film = PixelFilm(7, 7);
        const float qx = film.x / kFocalLength;
        const float qy = film.y / kFocalLength;
        float px = qx;
        float py = qy;
        for (int step = 0; step < 50; ++step) {
            const float scale = 1 + k1 * (px * px + py * py);
            px = qx / scale;
            py = qy / scale;
        }
        const float expected = kWallDistance * std::sqrt(1 + px * px + py * py);
        EXPECT_NEAR(PixelDepth(depth, 7, 7), expected, 0.02);
        EXPECT_GT(PixelDepth(depth, 7, 7),
                  kWallDistance * std::sqrt(1 + qx * qx + qy * qy) + 0.05f);
    });
}

TEST(CameraTest, SphericalSpansTheFullSphere) {
    ForEachDevice([](Device& device) {
        Renderer renderer(FacingWall(device, {.projection = Projection::Spherical}));
        const std::vector<float> depth = Distances(renderer);
        const float azimuth = std::numbers::pi_v<float> * 0.125f;
        const float elevation = std::numbers::pi_v<float> * 0.5f * -0.125f;
        EXPECT_NEAR(PixelDepth(depth, 4, 4),
                    kWallDistance / (std::cos(azimuth) * std::cos(elevation)), 0.02);
        EXPECT_TRUE(std::isinf(PixelDepth(depth, 0, 4)));
    });
}

TEST(CameraTest, PixelSamplesAverageCoverageAcrossThePixel) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        const float pixelWidth = kWallDistance * 2 * kHalfWidth / kSize / kFocalLength;
        const float edge = 0.3f * pixelWidth;
        scene->Define<Plane>(
            "/Wall", {.translate = {edge + 50, 0, -kWallDistance}, .width = 100, .length = 100});
        scene->Define<Camera>("/Cam", {.focalLength = kFocalLength});
        scene->Define<Scope>("/Render");
        scene->Define<RenderVar>("/Render/Color", {.sourceName = "HdrColor"});
        scene->Define<RenderProduct>("/Render/View", {.renderMode = RenderMode::RealTime,
                                                      .camera = {"/Cam"},
                                                      .resolution = {kSize, kSize},
                                                      .orderedVars = {"/Render/Color"}});
        Prim product = *scene->Find("/Render/View");
        product.Set(BackgroundAPI::kBackgroundColor, Float3{1, 1, 1});
        const auto coverage = [&](int32_t samples) {
            product.Set(RealTimeAPI::kPixelSamples, samples);
            Renderer renderer(scene);
            RenderOutput output = renderer.Step(kProducts, 0);
            output.Wait();
            const std::vector<float> color =
                output["/Render/View"].frames.at(0).Source("HdrColor").tensor.ToHost<float>();
            return color[(kSize / 2 * kSize + kSize / 2) * 4];
        };
        const float wall = coverage(1);
        ASSERT_LT(wall, 0.9f);
        EXPECT_NEAR(coverage(64), 0.3f + 0.7f * wall, 0.04);
    });
}

TEST(CameraTest, BandsFollowTheCamerasSpectralRange) {
    ForEachDevice([](Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        Prim wall = scene->Define<Plane>(
            "/Wall", {.translate = {0, 0, -kWallDistance}, .width = 100, .length = 100});
        Prim camera = scene->Define<Camera>("/Cam", {.focalLength = kFocalLength});
        scene->Define<Scope>("/Render");
        scene->Define<RenderVar>("/Render/Color", {.sourceName = "HdrColor"});
        Prim product =
            scene->Define<RenderProduct>("/Render/View", {.renderMode = RenderMode::RealTime,
                                                          .camera = {"/Cam"},
                                                          .resolution = {kSize, kSize},
                                                          .orderedVars = {"/Render/Color"}});
        Renderer renderer(scene);
        const auto center = [&](Float3 color, SpectralMode mode, Float2 range) {
            wall.Set(Gprim::kDisplayColor, color);
            product.Set(RenderProduct::kSpectralMode, mode);
            camera.Set(Camera::kDetectorSpectralRange, range);
            RenderOutput output = renderer.Step(kProducts, 0);
            output.Wait();
            const std::vector<float> pixels =
                output["/Render/View"].frames.at(0).Source("HdrColor").tensor.ToHost<float>();
            const uint32_t at = (kSize / 2 * kSize + kSize / 2) * 4;
            return Float3{pixels[at], pixels[at + 1], pixels[at + 2]};
        };
        const Float2 visible{380, 780};
        const Float2 infrared{700, 1000};
        const Float3 white = center({1, 1, 1}, SpectralMode::Rgb, visible);
        ASSERT_GT(white.x, 0.1f);
        const Float3 banded = center({1, 1, 1}, SpectralMode::Bands, visible);
        EXPECT_NEAR(banded.x, white.x, 1e-3f);
        EXPECT_NEAR(banded.y, white.y, 1e-3f);
        EXPECT_NEAR(banded.z, white.z, 1e-3f);
        EXPECT_GT(center({0, 0, 1}, SpectralMode::Bands, visible).z, 0.5f * white.z);
        const Float3 nearInfrared = center({1, 1, 1}, SpectralMode::Bands, infrared);
        EXPECT_NEAR(nearInfrared.x, white.x, 1e-3f);
        EXPECT_EQ(nearInfrared.y, 0);
        EXPECT_EQ(nearInfrared.z, 0);
        EXPECT_NEAR(center({1, 0, 0}, SpectralMode::Bands, infrared).x, white.x, 0.1f * white.x);
        EXPECT_LT(center({0, 0, 1}, SpectralMode::Bands, infrared).x, white.x);
    });
}

}  // namespace
}  // namespace ernie

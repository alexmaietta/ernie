// SPDX-License-Identifier: BSD-3-Clause
/// @file renderer_test.cpp
/// @brief The README's first render, as written there: a rock on the ground under the sun, seen
/// by a camera the render settings name.
#include "ernie/device/column.h"
#include "ernie/device/staging.h"
#include "ernie/render/products.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/schema/camera.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/lux.h"
#include "ernie/scene/schema/planet.h"
#include "ernie/scene/schema/render.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>

namespace ernie {
namespace {

/// @return A renderer, or none without a device that ray traces; no Vulkan at all fails where CI
/// requires it.
std::unique_ptr<Renderer> MakeRenderer() {
    const std::unique_ptr<DeviceContext> probe = test::MakeContext();
    if (!probe || !probe->Capabilities().rayQuery) {
        return nullptr;
    }
    return std::make_unique<Renderer>(RendererOptions{.validation = true});
}

/// @return The RGBA bytes at a pixel of an image `width` pixels across.
std::array<std::uint8_t, 4> Pixel(const Tensor& image, std::int64_t x, std::int64_t y) {
    const auto* bytes = static_cast<const std::uint8_t*>(image.Data());
    const std::int64_t at = 4 * (y * image.Shape()[1] + x);
    return {bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]};
}

TEST(Renderer, RendersTheReadmesFirstScene) {
    const std::unique_ptr<Renderer> renderer = MakeRenderer();
    if (!renderer) {
        GTEST_SKIP() << "No Vulkan device that ray traces";
    }
    // As the README's C++ example has it.
    Scene scene;
    GroundPlane::Define(scene, "/World/Ground");
    Sphere::Define(scene, "/World/Rock");
    SetXform(scene, "/World/Rock", {.translate = glm::dvec3(0.0, 0.0, 1.0)});  // 1 m up
    DistantLight::Define(scene, "/World/Sun");
    scene.Set("/World/Sun", tokens::kTransform, DistantLightTransform({0.4, -0.6, 0.7}));
    Camera::Define(scene, "/World/Camera");
    scene.Set("/World/Camera", tokens::kTransform,
              LookAtTransform({0.0, -8.0, 3.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}));
    RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution({640, 480});

    const Tensor image = renderer->Render(scene);
    ASSERT_EQ(image.Shape(), (std::vector<std::int64_t>{480, 640, 4}));
    EXPECT_EQ(image.Type(), DType::UInt8);
    // The camera looks 14 degrees down at the rock's centre, its view 17 degrees high: the ground
    // fills the rest, lit more steeply near the camera than far off.
    const std::array<std::uint8_t, 4> rock = Pixel(image, 320, 240);
    const std::array<std::uint8_t, 4> near = Pixel(image, 320, 470);
    const std::array<std::uint8_t, 4> far = Pixel(image, 320, 5);
    EXPECT_GT(rock[0], 100) << "the rock, lit";
    EXPECT_GT(near[0], far[0]) << "the ground, darker where the headlight grazes it";
    EXPECT_GT(far[0], 0) << "still the ground at the top";
    EXPECT_EQ(rock[3], 255);
    test::ExpectClean(renderer->Context());

    // Again, after the rock moves out of view: only what changed is redone.
    SetXform(scene, "/World/Rock", {.translate = glm::dvec3(50.0, 0.0, 1.0)});
    const std::array<std::uint8_t, 4> moved = Pixel(renderer->Render(scene), 320, 240);
    EXPECT_NE(moved, rock) << "the ground behind where the rock was";
    test::ExpectClean(renderer->Context());
}

/// @brief Defines a unit sphere 5 m ahead of cameras at the origin looking along +y.
void DefineSphereAhead(Scene& scene) {
    Sphere::Define(scene, "/World/Ball");
    SetXform(scene, "/World/Ball", {.translate = glm::dvec3(0.0, 5.0, 0.0)});
}

/// @brief Defines a camera at `eye` looking along +y, its film `aperture` metres across and down.
void DefineCamera(Scene& scene, const Path& path, const glm::dvec3& eye,
                  glm::dvec2 aperture = {0.02, 0.02}) {
    Camera::Define(scene, path).SetHorizontalAperture(aperture.x).SetVerticalAperture(aperture.y);
    scene.Set(path, tokens::kTransform,
              LookAtTransform(eye, eye + glm::dvec3(0.0, 1.0, 0.0), {0.0, 0.0, 1.0}));
}

/// @brief Defines a product through `camera` holding `vars`, each a RenderVar under it.
void DefineProduct(Scene& scene, const Path& path, const Path& camera, glm::dvec2 resolution,
                   const std::vector<std::string>& vars) {
    std::vector<Path> ordered;
    for (const std::string& var : vars) {
        ordered.emplace_back(path.String() + "/" + var);
        RenderVar::Define(scene, ordered.back()).SetSourceName(var);
    }
    RenderProduct::Define(scene, path)
        .SetCamera(camera)
        .SetResolution(resolution)
        .SetOrderedVars(ordered);
}

TEST(Renderer, StepsProductsAlikeInOneBatchAndOthersApart) {
    const std::unique_ptr<Renderer> renderer = MakeRenderer();
    if (!renderer) {
        GTEST_SKIP() << "No Vulkan device that ray traces";
    }
    Scene scene;
    DefineSphereAhead(scene);
    DefineCamera(scene, "/World/Near", {0.0, 0.0, 0.0});
    DefineCamera(scene, "/World/Far", {0.0, -5.0, 0.0});
    const std::vector<std::string> vars{std::string(render_vars::kDistanceToCamera),
                                        std::string(render_vars::kPrimId)};
    DefineProduct(scene, "/Render/Near", "/World/Near", {33, 33}, vars);
    DefineProduct(scene, "/Render/Far", "/World/Far", {33, 33}, vars);
    DefineProduct(scene, "/Render/Small", "/World/Near", {9, 9},
                  {std::string(render_vars::kNormal)});
    RenderSettings::Define(scene, "/Render/Settings")
        .SetProducts({Path("/Render/Near"), Path("/Render/Far"), Path("/Render/Small")});

    const std::span<const RenderBatch> batches = renderer->Step(scene);
    ASSERT_EQ(batches.size(), 2u);
    EXPECT_EQ(batches[0].products, (std::vector<Path>{Path("/Render/Near"), Path("/Render/Far")}));
    const RenderVarOutput* distances = batches[0].Find(render_vars::kDistanceToCamera);
    ASSERT_NE(distances, nullptr);
    EXPECT_EQ(distances->shape, (std::vector<std::int64_t>{2, 33, 33}));
    const Tensor depth = renderer->Read(*distances);
    const float* values = depth.As<float>();
    const std::size_t middle = 33 * 16 + 16;
    EXPECT_NEAR(values[middle], 4.0f, 1e-4f) << "the near view";
    EXPECT_NEAR(values[33 * 33 + middle], 9.0f, 1e-4f) << "the far view, stacked after it";
    const Tensor prims = renderer->Read(*batches[0].Find(render_vars::kPrimId));
    EXPECT_EQ(prims.Type(), DType::Int32);
    EXPECT_EQ(static_cast<std::uint32_t>(prims.As<std::int32_t>()[middle]),
              scene.Find("/World/Ball")->slot);
    EXPECT_EQ(prims.As<std::int32_t>()[0], -1) << "a corner sees nothing";
    const Tensor normals = renderer->Read(*batches[1].Find(render_vars::kNormal));
    EXPECT_EQ(normals.Shape(), (std::vector<std::int64_t>{1, 9, 9, 3}));
    EXPECT_NEAR(normals.As<float>()[3 * (9 * 4 + 4) + 1], -1.0f, 1e-4f) << "facing the camera";
    test::ExpectClean(renderer->Context());
}

TEST(Renderer, EachViewSeesItsWorldAndTheSharedPrims) {
    const std::unique_ptr<Renderer> renderer = MakeRenderer();
    if (!renderer) {
        GTEST_SKIP() << "No Vulkan device that ray traces";
    }
    // Three worlds at the same place, each with a ball nearer than the last, before a shared wall
    // 10 m out; each world's camera is under the world's root, as its ball is.
    Scene scene;
    Cube::Define(scene, "/Wall").SetSize(1.0);
    SetXform(scene, "/Wall",
             {.translate = glm::dvec3(0.0, 10.5, 0.0), .scale = glm::dvec3(20.0, 1.0, 20.0)});
    std::vector<Path> products;
    for (int world = 0; world < 3; ++world) {
        const std::string root = "/World" + std::to_string(world);
        Xform::Define(scene, root).SetWorld(world);
        Sphere::Define(scene, root + "/Ball");
        SetXform(scene, root + "/Ball", {.translate = glm::dvec3(0.0, 4.0 - world, 0.0)});
        DefineCamera(scene, root + "/Camera", {0.0, 0.0, 0.0});
        products.emplace_back("/Render/View" + std::to_string(world));
        DefineProduct(scene, products.back(), root + "/Camera", {9, 9},
                      {std::string(render_vars::kDistanceToCamera)});
    }
    RenderSettings::Define(scene, "/Render/Settings").SetProducts(products);

    const auto middles = [&] {
        const RenderBatch& batch = renderer->Step(scene).front();
        const Tensor depth = renderer->Read(*batch.Find(render_vars::kDistanceToCamera));
        std::vector<float> seen;
        for (std::int64_t view = 0; view < 3; ++view) {
            seen.push_back(depth.As<float>()[view * 81 + 40]);
        }
        return seen;
    };
    const std::vector<float> seen = middles();
    EXPECT_NEAR(seen[0], 3.0f, 1e-4f) << "world 0's ball, 4 m out";
    EXPECT_NEAR(seen[1], 2.0f, 1e-4f) << "world 1's, 3 m out";
    EXPECT_NEAR(seen[2], 1.0f, 1e-4f) << "world 2's, 2 m out";

    // World 1 moves into world 2 by its root, so its view sees past where its ball was to the
    // shared wall, and world 2's sees its own ball still.
    Xform(scene, "/World1").SetWorld(2);
    scene.Set("/World1/Camera", Imageable::kWorld, 1);
    const std::vector<float> moved = middles();
    EXPECT_NEAR(moved[1], 10.0f, 1e-4f) << "the shared wall";
    EXPECT_NEAR(moved[2], 1.0f, 1e-4f) << "nearer than world 1's ball, now in world 2";
    test::ExpectClean(renderer->Context());
}

TEST(Renderer, StepsRenderPosesWrittenOnTheDevice) {
    const std::unique_ptr<Renderer> renderer = MakeRenderer();
    if (!renderer) {
        GTEST_SKIP() << "No Vulkan device that ray traces";
    }
    Scene scene;
    DefineSphereAhead(scene);
    DefineCamera(scene, "/World/Camera", {0.0, 0.0, 0.0});
    DefineProduct(scene, "/Render/Product", "/World/Camera", {9, 9},
                  {std::string(render_vars::kDistanceToCamera)});
    RenderSettings::Define(scene, "/Render/Settings").SetProducts({Path("/Render/Product")});
    const DeviceColumn& translates =
        renderer->BindTransformOp(scene, "/World/Ball", "xformOp:translate");
    ASSERT_EQ(translates.Shape(), (std::vector<std::int64_t>{1, 3}));

    // As CUDA would, a write on the device the host never sees: the ball 2 m nearer.
    DeviceContext& context = renderer->Context();
    Uploader uploader(context, context.Main());
    const std::array<float, 3> nearer{0.0f, 3.0f, 0.0f};
    uploader.Write(translates.Storage(), translates.Range().offset, std::span<const float>(nearer));
    renderer->WaitFor(uploader.Flush());
    const RenderBatch& batch = renderer->Step(scene).front();
    const Tensor depth = renderer->Read(*batch.Find(render_vars::kDistanceToCamera));
    EXPECT_NEAR(depth.As<float>()[40], 2.0f, 1e-4f);
    test::ExpectClean(context);
}

TEST(Products, ConformTheFilmToTheImage) {
    Scene scene;
    DefineCamera(scene, "/World/Camera", {0.0, 0.0, 0.0}, {0.02, 0.01});
    const PrimId camera = *scene.Find("/World/Camera");
    // A 2:1 film on a square image: expanding keeps the width and grows the height to match.
    const Lens expanded = ReadLens(scene, camera, {100, 100}, 1.0, "expandAperture");
    EXPECT_FLOAT_EQ(expanded.focal.x, 0.05f * 100.0f / 0.02f);
    EXPECT_FLOAT_EQ(expanded.focal.y, expanded.focal.x);
    const Lens cropped = ReadLens(scene, camera, {100, 100}, 1.0, "cropAperture");
    EXPECT_FLOAT_EQ(cropped.focal.x, 0.05f * 100.0f / 0.01f);
    const Lens stretched = ReadLens(scene, camera, {100, 100}, 1.0, "adjustPixelAspectRatio");
    EXPECT_FLOAT_EQ(stretched.focal.y, 2.0f * stretched.focal.x);
    EXPECT_EQ(expanded.principal, glm::vec2(50.0f));
    // A film shifted right moves what the middle pixel sees right, so the principal point left.
    scene.Set(camera, Camera::kHorizontalApertureOffset, 0.002);
    EXPECT_FLOAT_EQ(ReadLens(scene, camera, {100, 100}, 1.0, "expandAperture").principal.x,
                    50.0f - 0.002f * 100.0f / 0.02f);
    ErnieLensAPI::Apply(scene, "/World/Camera").SetLensProjection("fisheye");
    EXPECT_THROW(ReadLens(scene, camera, {100, 100}, 1.0, "expandAperture"), std::invalid_argument)
        << "no silent pinhole for a fisheye";
}

TEST(Renderer, AsksForRenderSettingsWithACamera) {
    const std::unique_ptr<Renderer> renderer = MakeRenderer();
    if (!renderer) {
        GTEST_SKIP() << "No Vulkan device that ray traces";
    }
    Scene scene;
    EXPECT_THROW(renderer->Render(scene), std::invalid_argument);
    RenderSettings::Define(scene, "/Render/Settings");
    EXPECT_THROW(renderer->Render(scene), std::invalid_argument);
}

}  // namespace
}  // namespace ernie

// SPDX-License-Identifier: BSD-3-Clause
/// @file render_test.cpp
/// @brief Tests of render settings, products, vars and passes.
#include "ernie/scene/schema/collection.h"
#include "ernie/scene/schema/render.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace ernie {
namespace {

TEST(Render, SettingsProductsAndVarsHaveUsdsFallbacks) {
    Scene scene;
    RenderSettings settings = RenderSettings::Define(scene, "/Render/Settings");
    EXPECT_EQ(settings.GetResolution(), glm::dvec2(2048.0, 1080.0));
    EXPECT_EQ(settings.GetIncludedPurposes(), (std::vector<std::string>{"default", "render"}));
    EXPECT_EQ(RenderVar::Define(scene, "/Render/Vars/Color").GetDataType(), "color3f");
    EXPECT_EQ(RenderProduct::Define(scene, "/Render/Camera").GetProductType(), "raster");
}

TEST(Render, ErnieSettingsLetEachProductPickItsTier) {
    Scene scene;
    RenderSettings settings = RenderSettings::Define(scene, "/Render/Settings");
    RenderProduct hero = RenderProduct::Define(scene, "/Render/Hero");
    settings.SetCamera(Path("/World/Camera"))
        .SetResolution({640.0, 480.0})
        .SetProducts({Path("/Render/Hero")});
    EXPECT_EQ(settings.GetFidelity(), "realtime");
    hero.SetFidelity("predictive").SetSamples(1024).SetDenoiser("none").SetSpectralBands(32);
    EXPECT_TRUE(hero.HasFidelity()) << "authored on the product, over the settings'";
    EXPECT_FALSE(hero.HasCamera()) << "the settings' camera stands";
    EXPECT_EQ(hero.GetBounces(), -1) << "the tier's";
    EXPECT_THROW(hero.SetFidelity("cinematic"), std::invalid_argument);
}

TEST(Render, PassesSeeEverythingUnlessTheirCollectionsSayOtherwise) {
    Scene scene;
    const RenderPass pass = RenderPass::Define(scene, "/Render/Pass");
    EXPECT_TRUE(CollectionAPI(scene, "/Render/Pass", "renderVisibility").GetIncludeRoot());
    EXPECT_TRUE(CollectionAPI(scene, "/Render/Pass", "cameraVisibility").GetIncludeRoot());
    EXPECT_FALSE(scene.Extent(pass.Id()));
}

}  // namespace
}  // namespace ernie

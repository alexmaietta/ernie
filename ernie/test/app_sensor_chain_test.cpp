// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/sensor_chain.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using ernie::Path;
using ernie::Scene;
using ernie::SensorFilterRegistry;

/// The types of `product`'s chain as rendering resolves it.
std::vector<std::string> TypesOf(const Scene& scene,
                                 const Path& product) {
    std::vector<std::string> types;
    for (const ernie::SensorFilterStep& step :
         ernie::SensorFilterStepsOf(
             scene,
             ernie::RenderProduct(scene, product).GetFilters())) {
        types.push_back(step.type);
    }
    return types;
}

Scene CameraScene() {
    Scene scene;
    ernie::Camera::Define(scene, "/World/Thermal");
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Thermal");
    return scene;
}

}  // namespace

TEST(SensorChain, CustomizingGivesTheCameraTheDefaultChainToEdit) {
    Scene scene = CameraScene();
    const Path camera("/World/Thermal");
    EXPECT_TRUE(
        ernie::ViewChainProduct(scene, "/Render/Settings", camera)
            .IsEmpty());
    const Path product = ernie::app::CustomizeChain(scene, camera);
    EXPECT_EQ(product, Path("/Render/ThermalSensor"));
    EXPECT_EQ(
        ernie::ViewChainProduct(scene, "/Render/Settings", camera),
        product);
    EXPECT_EQ(TypesOf(scene, product),
              (std::vector<std::string>{"qe", "noise", "adc", "cie",
                                        "exposure"}));
    // A product the settings name looking through it comes first.
    ernie::RenderProduct::Define(scene, "/Render/Own")
        .SetCamera(camera);
    ernie::RenderSettings(scene, "/Render/Settings")
        .SetProducts({Path("/Render/Own")});
    EXPECT_EQ(
        ernie::ViewChainProduct(scene, "/Render/Settings", camera),
        Path("/Render/Own"));
    EXPECT_EQ(ernie::app::CustomizeChain(scene, camera),
              Path("/Render/ThermalSensor1"));
}

TEST(SensorChain, FiltersBypassMoveAddAndRemove) {
    Scene scene = CameraScene();
    const SensorFilterRegistry registry =
        SensorFilterRegistry::WithStandard();
    const Path product =
        ernie::app::CustomizeChain(scene, "/World/Thermal");
    const Path noise("/Render/ThermalSensorChain/noise");
    ernie::SensorFilterPrim(scene, noise).SetFilterEnabled(false);
    EXPECT_EQ(
        TypesOf(scene, product),
        (std::vector<std::string>{"qe", "adc", "cie", "exposure"}))
        << "bypassed";
    ernie::SensorFilterPrim(scene, noise).SetFilterEnabled(true);

    // Each where what it reads is made: the mosaic on the electrons
    // after the noise, white balance on the colour after cie, a tone
    // curve on the display colour at the end, a point spread on the
    // spectrum first.
    ernie::app::AddChainFilter(scene, product, "bayer", registry);
    ernie::app::AddChainFilter(scene, product, "whiteBalance",
                               registry);
    ernie::app::AddChainFilter(scene, product, "toneCurve", registry);
    ernie::app::AddChainFilter(scene, product, "psf", registry);
    EXPECT_EQ(TypesOf(scene, product),
              (std::vector<std::string>{"psf", "qe", "noise", "bayer",
                                        "adc", "cie", "whiteBalance",
                                        "exposure", "toneCurve"}));
    EXPECT_TRUE(scene.Has("/Render/ThermalSensorChain/bayer"));

    ernie::app::MoveChainFilter(scene, product, 3, -1);
    EXPECT_EQ(TypesOf(scene, product)[2], "bayer");
    ernie::app::MoveChainFilter(scene, product, 0, -1);  // stays
    EXPECT_EQ(TypesOf(scene, product)[0], "psf");

    ernie::app::RemoveChainFilter(scene, product, 2);
    EXPECT_FALSE(scene.Has("/Render/ThermalSensorChain/bayer"));
    EXPECT_EQ(TypesOf(scene, product).size(), 8u);

    EXPECT_EQ(ernie::app::ChainFlowOf("qe", registry),
              "spectrum > electrons");
    EXPECT_EQ(ernie::app::ChainFlowOf("bolometer", registry),
              "spectrum > electrons");
    EXPECT_EQ(ernie::app::ChainFlowOf("toneCurve", registry),
              "display");
    EXPECT_EQ(ernie::app::ChainFlowOf("nope", registry), "unknown");
}

TEST(SensorChain, ProductsHoldVarsAndTheViewersCameraItsOwn) {
    Scene scene = CameraScene();
    const SensorFilterRegistry registry =
        SensorFilterRegistry::WithStandard();
    const Path camera("/World/Thermal");
    EXPECT_TRUE(ernie::app::ProductsOf(scene, camera).empty());
    const Path product = ernie::app::CustomizeChain(scene, camera);
    EXPECT_EQ(ernie::app::ProductsOf(scene, camera),
              std::vector<Path>{product});
    EXPECT_TRUE(ernie::app::HoldsVar(scene, product, "LdrColor"));
    EXPECT_FALSE(ernie::app::HoldsVar(scene, product, "Temperature"));
    ernie::app::SetProductVar(scene, product, "Temperature", true);
    EXPECT_TRUE(ernie::app::HoldsVar(scene, product, "Temperature"));
    EXPECT_TRUE(scene.Has(product.Child("Temperature")));
    ernie::app::SetProductVar(scene, product, "LdrColor", false);
    EXPECT_FALSE(ernie::app::HoldsVar(scene, product, "LdrColor"));
    EXPECT_FALSE(scene.Has(product.Child("LdrColor")));

    // The viewer's own camera's product under the session root, which
    // only it sees.
    const Path session(
        Path(ernie::tokens::kSessionRoot).Child("View"));
    ernie::Camera::Define(scene, session);
    const Path own = ernie::app::CustomizeChain(scene, session);
    EXPECT_TRUE(own.HasPrefix(Path(ernie::tokens::kSessionRoot)));
    EXPECT_EQ(
        ernie::ViewChainProduct(scene, "/Render/Settings", session),
        own);
    EXPECT_EQ(ernie::app::ProductsOf(scene, session),
              std::vector<Path>{own});
    EXPECT_EQ(ernie::app::ProductsOf(scene, camera),
              std::vector<Path>{product});

    EXPECT_EQ(ernie::app::StageVarOf("noise", registry), "Electrons");
    EXPECT_EQ(ernie::app::StageVarOf("adc", registry), "RawDN");
    EXPECT_EQ(ernie::app::StageVarOf("cie", registry), "HdrColor");
    EXPECT_EQ(ernie::app::StageVarOf("psf", registry),
              "SpectralRadiance");
    EXPECT_EQ(ernie::app::StageVarOf("exposure", registry),
              "LdrColor");
}

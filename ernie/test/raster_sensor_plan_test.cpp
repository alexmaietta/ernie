// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/sensor_plan.h"

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

using ernie::PlanSensor;
using ernie::SensorFilter;
using namespace ernie::render_vars;

TEST(PlanSensor, RunsOnlyTheFiltersTheHeldVarsNeed) {
    const std::vector<SensorFilter> chain =
        ernie::DefaultFilterChain();
    const std::string_view colour[] = {kLdrColor};
    EXPECT_EQ(PlanSensor(chain, colour).filters,
              (std::vector<SensorFilter>{SensorFilter::Lens,
                                         SensorFilter::Cie,
                                         SensorFilter::Exposure}));
    const std::string_view raw[] = {kRawDn};
    EXPECT_EQ(PlanSensor(chain, raw).filters,
              (std::vector<SensorFilter>{
                  SensorFilter::Lens, SensorFilter::Qe,
                  SensorFilter::Noise, SensorFilter::Adc}));
    const std::string_view depth[] = {kDistanceToCamera};
    EXPECT_EQ(PlanSensor(chain, depth).filters,
              std::vector<SensorFilter>{SensorFilter::Lens});
    const std::string_view hdr[] = {kHdrColor};
    const SensorFilter cieOnly[] = {SensorFilter::Cie};
    EXPECT_EQ(PlanSensor(cieOnly, hdr).filters,
              (std::vector<SensorFilter>{SensorFilter::Lens,
                                         SensorFilter::Cie}))
        << "the lens heads every chain";
}

TEST(PlanSensor, FusesTheChainInOrderAndWritesOnlyHeldVars) {
    // Noise after the ADC: RawDN has no noise, Electrons does.
    const SensorFilter chain[] = {SensorFilter::Qe, SensorFilter::Adc,
                                  SensorFilter::Noise};
    const std::string_view held[] = {kElectrons, kRawDn};
    const ernie::SensorPlan plan = PlanSensor(chain, held);
    const std::string source = plan.Source();
    const auto at = [&](std::string_view text) {
        return source.find(text);
    };
    ASSERT_NE(at("Qe.accumulate"), std::string::npos);
    EXPECT_LT(at("Qe.finish"), at("Adc.apply"));
    EXPECT_LT(at("Adc.apply"), at("Noise.apply"));
    EXPECT_EQ(at("writeSpectral"), std::string::npos)
        << "SpectralRadiance is not held";
    EXPECT_EQ(at("Cie."), std::string::npos);
    EXPECT_EQ(plan.holds, (1u << 1) | (1u << 2))
        << "Electrons and RawDN, by kRenderVarNames";

    const std::string_view spectral[] = {kSpectralRadiance, kRawDn};
    EXPECT_NE(
        PlanSensor(chain, spectral).Source().find("writeSpectral"),
        std::string::npos);
    EXPECT_NE(PlanSensor(chain, spectral).Key(), plan.Key());
    EXPECT_EQ(PlanSensor(chain, held).Key(), plan.Key())
        << "the same plan shares its kernel";
}

TEST(PlanSensor, CompilesTheLensProjectionIn) {
    ernie::LensModel lens;
    EXPECT_EQ(ernie::LensKernelOf(lens), "Pinhole");
    lens.k1 = -0.1;
    EXPECT_EQ(ernie::LensKernelOf(lens), "BrownConrady");
    lens.projection = ernie::LensProjection::Fisheye;
    EXPECT_EQ(ernie::LensKernelOf(lens), "Fisheye");

    const std::string_view held[] = {kHdrColor};
    ernie::SensorPlan plan =
        PlanSensor(ernie::DefaultFilterChain(), held);
    const std::string pinhole = plan.Key();
    EXPECT_NE(plan.LensSource().find("Pinhole.pinholeOf"),
              std::string::npos);
    EXPECT_EQ(plan.LensSource().find("BrownConrady"),
              std::string::npos)
        << "a pinhole's kernel has no distortion";
    plan.lens = "Fisheye";
    EXPECT_NE(plan.Key(), pinhole)
        << "each projection has its own kernel";
    EXPECT_NE(plan.LensSource().find("Fisheye.falloff"),
              std::string::npos);
}

TEST(PlanSensor, RejectsChainsThatReadBeforeWriting) {
    const SensorFilter chain[] = {SensorFilter::Exposure};
    const std::string_view held[] = {kLdrColor};
    EXPECT_THROW(PlanSensor(chain, held), std::invalid_argument);
}

TEST(PlanSensor, CallsCustomFiltersWithTheirStepsInputs) {
    ernie::SensorFilterModel hot;
    hot.id = "hotPixels";
    hot.kind = ernie::FilterKind::Pixel;
    hot.domain = ernie::FilterDomain::Electrons;
    hot.params = {{"fraction", 1e-4f, ""},
                  {"electrons", 2000.0f, ""}};
    hot.typeName = "HotPixels";
    hot.slang = "import sensor; struct HotPixels : IPixelFilter { /* "
                "... */ }";
    ernie::SensorFilterRegistry registry;
    registry.Add(hot);

    const ernie::SensorFilterStep chain[] = {
        {"qe", {}},
        {"hotPixels", {{"electrons", 500.0f}}},
        {"noise", {}},
        {"adc", {}}};
    const std::string_view raw[] = {kRawDn};
    const ernie::SensorPlan plan = PlanSensor(chain, raw, registry);
    EXPECT_EQ(plan.params, (std::vector<float>{1e-4f, 500.0f}))
        << "defaults, then inputs";
    const std::string source = plan.Source();
    EXPECT_NE(source.find("import filter_hotPixels;"),
              std::string::npos);
    EXPECT_NE(source.find("params2.electrons = filterParams[1];"),
              std::string::npos);
    EXPECT_LT(source.find("HotPixels.apply(value, pixel, params2)"),
              source.find("Noise.apply"));
    ASSERT_EQ(plan.Modules().size(), 1u);
    EXPECT_EQ(plan.Modules().front().first, "filter_hotPixels");
    EXPECT_NE(plan.Key().find("hotPixels"), std::string::npos);

    const std::string_view colour[] = {kLdrColor};
    EXPECT_TRUE(PlanSensor(chain, colour, registry).params.empty())
        << "a filter on electrons does not run for colour alone";
}

namespace {

ernie::SensorFilterModel Spatial(std::string id,
                                 ernie::FilterDomain domain) {
    ernie::SensorFilterModel model;
    model.id = std::move(id);
    model.kind = ernie::FilterKind::Spatial;
    model.domain = domain;
    model.typeName = "Spread";
    model.slang = "import sensor; struct Spread {}";
    return model;
}

}  // namespace

TEST(PlanSensor, SplitsTheKernelAtSpatialFiltersOnTheirBranch) {
    ernie::SensorFilterRegistry registry;
    registry.Add(Spatial("glow", ernie::FilterDomain::Rgb));
    registry.Add(Spatial("blur", ernie::FilterDomain::Spectrum));

    // Glare on linear colour splits the colour branch only: RawDN is
    // final in the first kernel.
    const ernie::SensorFilterStep glare[] = {{"qe", {}},
                                             {"adc", {}},
                                             {"cie", {}},
                                             {"glow", {}},
                                             {"exposure", {}}};
    const std::string_view rawAndColour[] = {kRawDn, kLdrColor};
    const ernie::SensorPlan split =
        PlanSensor(glare, rawAndColour, registry);
    ASSERT_EQ(split.segments, 2u);
    EXPECT_TRUE(split.channelImages);
    EXPECT_FALSE(split.spectrumImages);
    std::vector<std::uint32_t> segments;
    for (const ernie::SensorPlan::Stage& stage : split.stages) {
        segments.push_back(stage.segment);
    }
    EXPECT_EQ(segments,
              (std::vector<std::uint32_t>{0, 0, 0, 0, 1, 1}))
        << "lens, qe, adc, cie, then glow and exposure";
    const std::string first = split.Source(0);
    const std::string second = split.Source(1);
    EXPECT_NE(first.find("storeChannel(value, pixel, kChannelRgb)"),
              std::string::npos);
    EXPECT_NE(first.find("writeOutputs(value, pixel, 4u)"),
              std::string::npos)
        << "RawDN";
    EXPECT_NE(
        second.find("Spread.apply(value, pixel, input4, params4)"),
        std::string::npos);
    EXPECT_NE(second.find("writeOutputs(value, pixel, 16u)"),
              std::string::npos)
        << "LdrColor";
    EXPECT_EQ(second.find("pixel.bands()"), std::string::npos)
        << "no spectrum after the split";

    // A blur on the spectrum moves everything after it, both
    // branches, to the second kernel.
    const ernie::SensorFilterStep psf[] = {
        {"blur", {}}, {"qe", {}}, {"cie", {}}};
    const std::string_view held[] = {kSpectralRadiance, kElectrons,
                                     kHdrColor};
    const ernie::SensorPlan blurred = PlanSensor(psf, held, registry);
    ASSERT_EQ(blurred.segments, 2u);
    EXPECT_TRUE(blurred.spectrumImages);
    const std::string before = blurred.Source(0);
    const std::string after = blurred.Source(1);
    EXPECT_NE(before.find("storeSpectrum"), std::string::npos);
    EXPECT_EQ(before.find("writeSpectral"), std::string::npos);
    EXPECT_NE(after.find("Spread.band(band, pixel, input1, params1)"),
              std::string::npos);
    EXPECT_NE(after.find("writeSpectral"), std::string::npos);
    EXPECT_NE(after.find("Qe.accumulate"), std::string::npos);

    // Two in a row alternate the images between three kernels.
    const ernie::SensorFilterStep twice[] = {
        {"cie", {}}, {"glow", {}}, {"glow", {}}};
    const std::string_view hdr[] = {kHdrColor};
    EXPECT_EQ(PlanSensor(twice, hdr, registry).segments, 3u);
}

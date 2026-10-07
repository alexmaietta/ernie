#include "ernie/raster/sensor_plan.h"

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

using ernie::PlanSensor;
using ernie::SensorFilter;
using namespace ernie::render_vars;

TEST(PlanSensor, RunsOnlyTheFiltersTheHeldVarsNeed) {
    const std::vector<SensorFilter> chain = ernie::DefaultFilterChain();
    const std::string_view colour[] = {kLdrColor};
    EXPECT_EQ(
        PlanSensor(chain, colour).filters,
        (std::vector<SensorFilter>{SensorFilter::Lens, SensorFilter::Cie, SensorFilter::Exposure}));
    const std::string_view raw[] = {kRawDn};
    EXPECT_EQ(PlanSensor(chain, raw).filters,
              (std::vector<SensorFilter>{SensorFilter::Lens, SensorFilter::Qe, SensorFilter::Noise,
                                         SensorFilter::Adc}));
    const std::string_view depth[] = {kDistanceToCamera};
    EXPECT_EQ(PlanSensor(chain, depth).filters, std::vector<SensorFilter>{SensorFilter::Lens});
    const std::string_view hdr[] = {kHdrColor};
    const SensorFilter cieOnly[] = {SensorFilter::Cie};
    EXPECT_EQ(PlanSensor(cieOnly, hdr).filters,
              (std::vector<SensorFilter>{SensorFilter::Lens, SensorFilter::Cie}))
        << "the lens heads every chain";
}

TEST(PlanSensor, FusesTheChainInOrderAndWritesOnlyHeldVars) {
    // Noise after the ADC: RawDN has no noise, Electrons does.
    const SensorFilter chain[] = {SensorFilter::Qe, SensorFilter::Adc, SensorFilter::Noise};
    const std::string_view held[] = {kElectrons, kRawDn};
    const ernie::SensorPlan plan = PlanSensor(chain, held);
    const std::string source = plan.Source();
    const auto at = [&](std::string_view text) { return source.find(text); };
    ASSERT_NE(at("Qe.accumulate"), std::string::npos);
    EXPECT_LT(at("Qe.finish"), at("Adc.apply"));
    EXPECT_LT(at("Adc.apply"), at("Noise.apply"));
    EXPECT_EQ(at("writeSpectral"), std::string::npos) << "SpectralRadiance is not held";
    EXPECT_EQ(at("Cie."), std::string::npos);
    EXPECT_EQ(plan.holds, (1u << 1) | (1u << 2)) << "Electrons and RawDN, by kRenderVarNames";

    const std::string_view spectral[] = {kSpectralRadiance, kRawDn};
    EXPECT_NE(PlanSensor(chain, spectral).Source().find("writeSpectral"), std::string::npos);
    EXPECT_NE(PlanSensor(chain, spectral).Key(), plan.Key());
    EXPECT_EQ(PlanSensor(chain, held).Key(), plan.Key()) << "the same plan shares its kernel";
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
    hot.params = {{"fraction", 1e-4f, ""}, {"electrons", 2000.0f, ""}};
    hot.typeName = "HotPixels";
    hot.slang = "import sensor; struct HotPixels : IPixelFilter { /* ... */ }";
    ernie::SensorFilterRegistry registry;
    registry.Add(hot);

    const ernie::SensorFilterStep chain[] = {
        {"qe", {}}, {"hotPixels", {{"electrons", 500.0f}}}, {"noise", {}}, {"adc", {}}};
    const std::string_view raw[] = {kRawDn};
    const ernie::SensorPlan plan = PlanSensor(chain, raw, registry);
    EXPECT_EQ(plan.params, (std::vector<float>{1e-4f, 500.0f})) << "defaults, then inputs";
    const std::string source = plan.Source();
    EXPECT_NE(source.find("import filter_hotPixels;"), std::string::npos);
    EXPECT_NE(source.find("params2.electrons = filterParams[1];"), std::string::npos);
    EXPECT_LT(source.find("HotPixels.apply(value, pixel, params2)"), source.find("Noise.apply"));
    ASSERT_EQ(plan.Modules().size(), 1u);
    EXPECT_EQ(plan.Modules().front().first, "filter_hotPixels");
    EXPECT_NE(plan.Key().find("hotPixels"), std::string::npos);

    const std::string_view colour[] = {kLdrColor};
    EXPECT_TRUE(PlanSensor(chain, colour, registry).params.empty())
        << "a filter on electrons does not run for colour alone";
}

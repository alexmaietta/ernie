// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/sensor_filters.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <vector>

using ernie::SensorFilter;
namespace render_vars = ernie::render_vars;

namespace {

bool Contains(const std::vector<std::string_view>& names,
              std::string_view name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

}  // namespace

TEST(SensorFilters, CurvesResampleLinearlyEndToEnd) {
    const std::vector<float> ramp = {0.0f, 1.0f};
    EXPECT_EQ(ernie::ResampleCurve(ramp, 5),
              (std::vector<float>{0.0f, 0.25f, 0.5f, 0.75f, 1.0f}));
    const std::vector<float> one = {0.5f};
    EXPECT_EQ(ernie::ResampleCurve(one, 3),
              (std::vector<float>{0.5f, 0.5f, 0.5f}));
    const std::vector<float> three = {0.0f, 2.0f, 0.0f};
    EXPECT_EQ(ernie::ResampleCurve(three, 5),
              (std::vector<float>{0.0f, 1.0f, 2.0f, 1.0f, 0.0f}));
    EXPECT_EQ(ernie::ResampleCurve({}, 2),
              (std::vector<float>{0.0f, 0.0f}));
}

TEST(SensorFilters,
     StandardFiltersRegisterAndTheirAovsLayOutInChainOrder) {
    const ernie::SensorFilterRegistry registry =
        ernie::SensorFilterRegistry::WithStandard();
    const std::vector<ernie::SensorFilterStep> chain = {
        {"qe", {}}, {"blooming", {}}, {"adc", {}}, {"demosaic", {}}};
    EXPECT_EQ(ernie::AovOutputsOf(chain, registry),
              (std::vector<std::string>{"overflow", "mosaic"}));
    ernie::SensorFilterModel huge =
        ernie::StandardSensorFilters().front();
    huge.id = "huge";
    huge.params = {{"curve", 0.0f, "", 2000}};
    ernie::SensorFilterRegistry copy = registry;
    EXPECT_THROW(copy.Add(huge), std::invalid_argument)
        << "more floats than kMaxParamFloats";
}

TEST(SensorFilters, NamesRoundTrip) {
    for (const SensorFilter filter : ernie::kSensorFilters) {
        EXPECT_EQ(ernie::SensorFilterNamed(ernie::NameOf(filter)),
                  filter);
    }
    EXPECT_FALSE(ernie::SensorFilterNamed("bloom").has_value());
    EXPECT_EQ(ernie::DefaultFilterChain().size(),
              ernie::kSensorFilters.size());
}

TEST(SensorFilters, CheckFilterChainPrependsTheLensAndChecksOrder) {
    const std::vector<SensorFilter> cie =
        ernie::CheckFilterChain(std::vector<SensorFilter>{
            SensorFilter::Cie, SensorFilter::Exposure});
    EXPECT_EQ(cie, (std::vector<SensorFilter>{
                       SensorFilter::Lens, SensorFilter::Cie,
                       SensorFilter::Exposure}));
    EXPECT_EQ(ernie::CheckFilterChain(ernie::DefaultFilterChain()),
              ernie::DefaultFilterChain());
    EXPECT_EQ(ernie::CheckFilterChain({}),
              std::vector<SensorFilter>{SensorFilter::Lens});

    EXPECT_THROW(ernie::CheckFilterChain(
                     std::vector<SensorFilter>{SensorFilter::Noise}),
                 std::invalid_argument)
        << "noise needs electrons";
    EXPECT_THROW(ernie::CheckFilterChain(
                     std::vector<SensorFilter>{SensorFilter::Adc}),
                 std::invalid_argument);
    EXPECT_THROW(ernie::CheckFilterChain(std::vector<SensorFilter>{
                     SensorFilter::Exposure}),
                 std::invalid_argument)
        << "exposure needs HdrColor";
    EXPECT_THROW(ernie::CheckFilterChain(std::vector<SensorFilter>{
                     SensorFilter::Qe, SensorFilter::Qe}),
                 std::invalid_argument);
}

TEST(SensorFilters, VarsWrittenByFollowTheChain) {
    const std::vector<std::string_view> lens = ernie::VarsWrittenBy(
        std::vector<SensorFilter>{SensorFilter::Lens});
    for (const std::string_view name :
         {render_vars::kSpectralRadiance,
          render_vars::kDistanceToImagePlane,
          render_vars::kDistanceToCamera, render_vars::kNormal,
          render_vars::kPrimId, render_vars::kInstanceId}) {
        EXPECT_TRUE(Contains(lens, name)) << name;
    }
    EXPECT_FALSE(Contains(lens, render_vars::kLdrColor));

    const std::vector<std::string_view> all =
        ernie::VarsWrittenBy(ernie::DefaultFilterChain());
    EXPECT_EQ(all.size(), ernie::kRenderVarNames.size())
        << "the default chain writes every var";
    for (const std::string_view name : ernie::kRenderVarNames) {
        EXPECT_TRUE(Contains(all, name)) << name;
    }
}

TEST(RequiredVars, HoldsJustTheRequestedVarsInBindingOrder) {
    using namespace ernie::render_vars;
    const std::string_view colour[] = {kLdrColor, kDistanceToCamera};
    EXPECT_EQ(
        ernie::RequiredVars(colour),
        (std::vector<std::string_view>{kLdrColor, kDistanceToCamera}))
        << "what LdrColor is made from stays in the sensor kernel's "
           "registers";
    const std::string_view raw[] = {kRawDn, kSpectralRadiance};
    EXPECT_EQ(
        ernie::RequiredVars(raw),
        (std::vector<std::string_view>{kSpectralRadiance, kRawDn}));
    const std::vector<std::string_view> every =
        ernie::RequiredVars({});
    EXPECT_EQ(every.size(), ernie::kRenderVarNames.size() - 1)
        << "empty is all but one";
    EXPECT_EQ(std::count(every.begin(), every.end(),
                         ernie::render_vars::kMotionVector),
              0)
        << "MotionVector is opt-in";
}

namespace {

/// A custom pixel filter on linear colour scaling it by `scale`.
ernie::SensorFilterModel Gain() {
    ernie::SensorFilterModel model;
    model.id = "gain";
    model.kind = ernie::FilterKind::Pixel;
    model.domain = ernie::FilterDomain::Rgb;
    model.params = {{"scale", 2.0f, "colour multiplier"}};
    model.typeName = "Gain";
    model.slang = R"(import sensor;
struct Gain : IPixelFilter {
    struct Params { float scale; }
    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        value.rgb *= params.scale;
    }
})";
    return model;
}

ernie::SensorFilterStep Step(std::string type) {
    return {std::move(type), {}};
}

}  // namespace

TEST(SensorFilterRegistry, ChecksCustomFilters) {
    ernie::SensorFilterRegistry registry;
    registry.Add(Gain());
    ASSERT_NE(registry.Find("gain"), nullptr);
    EXPECT_EQ(registry.Find("gain")->params.front().defaultValue,
              2.0f);
    EXPECT_THROW(registry.Add(Gain()), std::invalid_argument)
        << "taken";

    const auto rejects = [&registry](auto change) {
        ernie::SensorFilterModel model = Gain();
        model.id = "other";
        change(model);
        EXPECT_THROW(registry.Add(model), std::invalid_argument);
    };
    rejects([](ernie::SensorFilterModel& m) {
        m.id = "noise";
    });  // a built-in's name
    rejects([](ernie::SensorFilterModel& m) { m.id = "not an id"; });
    rejects(
        [](ernie::SensorFilterModel&
               m) {  // integrators make electrons or linear colour
            m.kind = ernie::FilterKind::Integrator;
            m.domain = ernie::FilterDomain::Display;
        });
    rejects([](ernie::SensorFilterModel& m) {
        m.domain = ernie::FilterDomain::Spectrum;
    });
    rejects([](ernie::SensorFilterModel& m) {
        m.kind = ernie::FilterKind::Band;
    });  // on rgb
    rejects([](ernie::SensorFilterModel& m) {
        m.params.push_back(m.params.front());
    });
    rejects([](ernie::SensorFilterModel& m) {
        m.params.resize(17, {"p", 0.0f, ""});
    });
    rejects(
        [](ernie::SensorFilterModel& m) { m.typeName = "Missing"; });
}

TEST(CheckFilterChain, PlacesCustomFiltersAfterWhatTheyWorkOn) {
    ernie::SensorFilterRegistry registry;
    registry.Add(Gain());
    const ernie::SensorFilterStep chain[] = {
        Step("cie"), Step("gain"), Step("exposure")};
    const std::vector<ernie::SensorFilterStep> checked =
        ernie::CheckFilterChain(chain, registry);
    ASSERT_EQ(checked.size(), 4u);
    EXPECT_EQ(checked.front().type, "lens")
        << "the lens is prepended";

    const ernie::SensorFilterStep early[] = {Step("gain"),
                                             Step("cie")};
    EXPECT_THROW(ernie::CheckFilterChain(early, registry),
                 std::invalid_argument)
        << "no linear colour before CIE";
    const ernie::SensorFilterStep unknown[] = {Step("cie"),
                                               Step("blur")};
    EXPECT_THROW(ernie::CheckFilterChain(unknown, registry),
                 std::invalid_argument);
    const ernie::SensorFilterStep lensLater[] = {Step("cie"),
                                                 Step("lens")};
    EXPECT_THROW(ernie::CheckFilterChain(lensLater, registry),
                 std::invalid_argument);

    ernie::SensorFilterModel band = Gain();
    band.id = "irCut";
    band.kind = ernie::FilterKind::Band;
    band.domain = ernie::FilterDomain::Spectrum;
    registry.Add(band);
    const ernie::SensorFilterStep afterQe[] = {Step("qe"),
                                               Step("irCut")};
    EXPECT_THROW(ernie::CheckFilterChain(afterQe, registry),
                 std::invalid_argument)
        << "band filters come before the integrators";
    const ernie::SensorFilterStep beforeQe[] = {Step("irCut"),
                                                Step("qe")};
    EXPECT_NO_THROW(ernie::CheckFilterChain(beforeQe, registry));
}

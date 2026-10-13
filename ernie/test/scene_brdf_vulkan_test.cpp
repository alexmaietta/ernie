// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/buffer.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/scene/brdf.h"
#include "support/brdf_reference.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using ernie::BrdfModel;
using ernie::BrdfRegistry;
using ernie::BufferMemory;
using ernie::ComputePipeline;
using ernie::GpuBuffer;
using ernie::ShaderCompiler;
using ernie::ShaderModuleSource;
using ernie::testing::Geometry;
using ernie::testing::GeometryGrid;
using ernie::testing::HapkeReference;
using ernie::testing::HenyeyGreenstein;
using ernie::testing::kPi;
using ernie::testing::PreviewSurfaceReference;

namespace {

/// Evaluates a model for every case: geometry is (mu0, mu, cos g, cos
/// psi), albedo in rgb, the unresolved slope variance in a.
constexpr char kHarness[] = R"(import brdf;
import MODULE;

struct Case {
    float4 geometry;
    float4 albedo;
}

[[vk::binding(0, 0)]] StructuredBuffer<Case> cases;
[[vk::binding(1, 0)]] StructuredBuffer<BrdfParams> params;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> results;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint count;
    uint stride;
    cases.GetDimensions(count, stride);
    if (id.x >= count) {
        return;
    }
    let c = cases[id.x];
    BrdfInput s;
    s.mu0 = c.geometry.x;
    s.mu = c.geometry.y;
    s.cosG = c.geometry.z;
    s.cosPsi = c.geometry.w;
    s.albedo = c.albedo.rgb;
    s.slopeVariance = c.albedo.a;
    // The light in the plane at azimuth ROTATION from the tangent, the viewer psi further round.
    let sinI = sqrt(max(1.0 - s.mu0 * s.mu0, 0.0));
    let sinE = sqrt(max(1.0 - s.mu * s.mu, 0.0));
    let psi = acos(clamp(s.cosPsi, -1.0, 1.0));
    s.light = float3(sinI * cos(ROTATION), sinI * sin(ROTATION), s.mu0);
    s.view = float3(sinE * cos(ROTATION + psi), sinE * sin(ROTATION + psi), s.mu);
    // The RGB slots as bands of these wavelengths, the model's colours their RGB.
    s.wavelengths = float3(610.0, 550.0, 465.0);
    for (uint k = 0; k < kBrdfColours; ++k) {
        s.colours[k] = float3(1.0);
    }
    COLOURS
    TYPE model;
    results[id.x] = float4(MODE, 0.0);
}
)";

class BrdfVulkanTest : public ernie::testing::VulkanTest {
  protected:
    /// A `BrdfParams` (its 16 core values, then the address of the
    /// rest after the material's features, padded to its stride) and
    /// the buffer it points into.
    struct ParamBuffers {
        std::unique_ptr<GpuBuffer> core;
        std::unique_ptr<GpuBuffer> extended;
    };

    ParamBuffers PackParams(const BrdfModel& model,
                            const std::vector<float>& values) {
        ParamBuffers buffers;
        std::vector<float> packed(20, 0.0f);
        std::copy_n(values.begin(),
                    std::min<std::size_t>(values.size(), 16),
                    packed.begin());
        if (values.size() > 16) {
            std::vector<float> rest = {std::bit_cast<float>(
                model.features ? model.features(values) : 0u)};
            rest.insert(rest.end(), values.begin() + 16,
                        values.end());
            buffers.extended = std::make_unique<GpuBuffer>(
                *context_, rest.size() * sizeof(float),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
            buffers.extended->Write(std::as_bytes(std::span(rest)));
            const std::uint64_t address = buffers.extended->Address();
            std::memcpy(&packed[16], &address, sizeof(address));
        }
        buffers.core = std::make_unique<GpuBuffer>(
            *context_, packed.size() * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
        buffers.core->Write(std::as_bytes(std::span(packed)));
        return buffers;
    }

    /// Evaluates `id`'s `radianceFactor` on the GPU for every
    /// geometry, with unresolved slopes of per-axis variance
    /// `slopeVariance`.
    ///
    /// `mode` is what it returns: the radiance factor, or with
    /// "specular" its specular part, or with "environment" the
    /// environment response's diffuse plus its specular lobes for a
    /// viewer at cos(e) mu. `features`, when given, is the
    /// BRDF_FEATURES the model is compiled with (a renderer's union
    /// over its materials). The surface's frame is turned `rotation`
    /// radians about its normal from the plane of incidence.
    std::vector<glm::dvec3>
    Evaluate(const std::string& id, const std::vector<float>& values,
             const std::vector<Geometry>& cases,
             double slopeVariance = 0.0,
             const std::string& mode = "radiance",
             std::optional<std::uint32_t> features = std::nullopt,
             double rotation = 0.0) {
        const BrdfModel* model = registry_.Find(id);
        if (model == nullptr) {
            throw std::invalid_argument(id);
        }
        std::string harness = kHarness;
        harness.replace(harness.find("MODULE"), 6,
                        ernie::BrdfModuleName(*model));
        harness.replace(harness.find("TYPE"), 4, model->typeName);
        std::string colours;
        const std::vector<std::size_t> reds = model->Colours();
        for (std::size_t c = 0; c < reds.size(); ++c) {
            colours += "s.colours[" + std::to_string(c) +
                       "] = float3(params[0].get(" +
                       std::to_string(reds[c]) + "), params[0].get(" +
                       std::to_string(reds[c] + 1) +
                       "), params[0].get(" +
                       std::to_string(reds[c] + 2) + "));\n    ";
        }
        harness.replace(harness.find("COLOURS"), 7, colours);
        for (std::size_t at = harness.find("ROTATION");
             at != std::string::npos; at = harness.find("ROTATION")) {
            harness.replace(at, 8, std::to_string(rotation));
        }
        const std::string evaluation =
            mode == "specular"
                ? "model.specularReflection(s, params[0]).factor"
            : mode == "subsurface" ? "model.subsurface(s, params[0])"
            : mode == "environment"
                ? "environmentSum(model.environment(s, params[0]))"
                : "model.radianceFactor(s, params[0])";
        harness.replace(harness.find("MODE"), 4, evaluation);
        harness += R"(
float3 environmentSum(EnvironmentResponse r) {
    return r.diffuse + r.specular0 + r.specular1;
}
)";
        std::vector<ShaderModuleSource> modules =
            ernie::BrdfModules(*model);
        if (features) {
            modules[1].source = "#define BRDF_FEATURES " +
                                std::to_string(*features) + "u\n" +
                                modules[1].source;
        }
        modules.push_back({"harness", harness});
        const ComputePipeline pipeline(
            *context_, compiler_.CompileSpirv(modules, "main"), 3);

        std::vector<float> packedCases;
        for (const Geometry& g : cases) {
            for (const double v :
                 {g.mu0, g.mu, g.cosG, g.cosPsi, g.albedo.r,
                  g.albedo.g, g.albedo.b, slopeVariance}) {
                packedCases.push_back(static_cast<float>(v));
            }
        }
        const ParamBuffers params = PackParams(*model, values);
        GpuBuffer caseBuffer(
            *context_, packedCases.size() * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
        const GpuBuffer& paramBuffer = *params.core;
        GpuBuffer resultBuffer(*context_,
                               cases.size() * 4 * sizeof(float),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               BufferMemory::Readback);
        caseBuffer.Write(std::as_bytes(std::span(packedCases)));
        const std::array<const GpuBuffer*, 3> buffers = {
            &caseBuffer, &paramBuffer, &resultBuffer};
        context_->Submit([&](VkCommandBuffer commands) {
            pipeline.Dispatch(
                commands, buffers,
                static_cast<std::uint32_t>((cases.size() + 63) / 64));
        });
        std::vector<float> raw(cases.size() * 4);
        resultBuffer.Read(std::as_writable_bytes(std::span(raw)));
        std::vector<glm::dvec3> results;
        for (std::size_t c = 0; c < cases.size(); ++c) {
            results.emplace_back(raw[4 * c], raw[4 * c + 1],
                                 raw[4 * c + 2]);
        }
        return results;
    }

    /// Compares GPU results with a per-channel reference within a
    /// relative tolerance.
    void ExpectMatches(
        const std::string& id, const std::vector<float>& params,
        const std::function<double(const Geometry&, double albedo)>&
            reference,
        double tolerance = 2e-4) {
        const std::vector<Geometry> cases = GeometryGrid();
        const std::vector<glm::dvec3> gpu =
            Evaluate(id, params, cases);
        for (std::size_t c = 0; c < cases.size(); ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                const double expected =
                    reference(cases[c], cases[c].albedo[ch]);
                EXPECT_NEAR(gpu[c][ch], expected,
                            tolerance *
                                std::max(1.0, std::abs(expected)))
                    << id << " case " << c << " channel " << ch
                    << " mu0 " << cases[c].mu0 << " mu "
                    << cases[c].mu << " cos_psi " << cases[c].cosPsi;
            }
        }
    }

    BrdfRegistry registry_ = BrdfRegistry::WithBuiltins();
    ShaderCompiler compiler_;
};

}  // namespace

TEST_F(BrdfVulkanTest, LambertIsAlbedoTimesCosineIncidence) {
    ExpectMatches("lambert", {}, [](const Geometry& g, double a) {
        return a * g.mu0;
    });
}

TEST_F(BrdfVulkanTest, LommelSeeligerMatchesItsClosedForm) {
    for (const float xi : {0.0f, -0.4f, 0.3f}) {
        ExpectMatches("lommel_seeliger", {xi},
                      [xi](const Geometry& g, double w) {
                          return w / 4.0 * g.mu0 / (g.mu0 + g.mu) *
                                 HenyeyGreenstein(xi, g.cosG);
                      });
    }
}

TEST_F(BrdfVulkanTest,
       LunarLambertBlendsLommelSeeligerAndLambertByPhase) {
    const std::vector<float> params = {-0.019f, 0.000242f,
                                       -0.00000146f};
    const std::vector<double> c(params.begin(), params.end());
    ExpectMatches("lunar_lambert", params,
                  [&c](const Geometry& g, double a) {
                      const double phase =
                          std::acos(std::clamp(g.cosG, -1.0, 1.0)) *
                          180.0 / kPi;
                      const double l = 1.0 + c[0] * phase +
                                       c[1] * phase * phase +
                                       c[2] * phase * phase * phase;
                      return a * (2.0 * l * g.mu0 / (g.mu0 + g.mu) +
                                  (1.0 - l) * g.mu0);
                  });
}

TEST_F(BrdfVulkanTest,
       OrenNayarMatchesTheQualitativeModelAndReducesToLambert) {
    const double sigma = 25.0 * kPi / 180.0;
    ExpectMatches("oren_nayar", {25.0f},
                  [sigma](const Geometry& g, double rho) {
                      const double s2 = sigma * sigma;
                      const double a = 1.0 - 0.5 * s2 / (s2 + 0.33);
                      const double b = 0.45 * s2 / (s2 + 0.09);
                      const double i = std::acos(g.mu0);
                      const double e = std::acos(g.mu);
                      return rho * g.mu0 *
                             (a + b * std::max(0.0, g.cosPsi) *
                                      std::sin(std::max(i, e)) *
                                      std::tan(std::min(i, e)));
                  });
    ExpectMatches(
        "oren_nayar", {0.0f},
        [](const Geometry& g, double rho) { return rho * g.mu0; });
}

TEST_F(BrdfVulkanTest, HapkeMatchesTheReferenceImplementation) {
    const std::vector<std::vector<float>> parameterSets = {
        BrdfRegistry::WithBuiltins().Find("hapke")->Defaults(),
        {0.3f, -0.2f, 2.5f, 0.02f, 35.0f, 0.6f},
        {0.1f, 0.9f, 0.0f, 0.1f, 0.0f, 0.0f}};
    for (const auto& params : parameterSets) {
        ExpectMatches(
            "hapke", params,
            [&params](const Geometry& g, double w) {
                return HapkeReference(g, w, params);
            },
            1e-3);
    }
}

TEST_F(BrdfVulkanTest, UnresolvedSlopesWidenTheModelsRoughness) {
    // Slopes of per-axis variance v shade as Hapke's theta-bar
    // widened to atan(sqrt(tan^2 theta-bar + 2 v / pi)), and as
    // Oren-Nayar's sigma widened by atan(sqrt(v)).
    const std::vector<Geometry> cases = GeometryGrid();
    const double variance = 0.05;
    std::vector<float> hapke =
        BrdfRegistry::WithBuiltins().Find("hapke")->Defaults();
    const std::vector<glm::dvec3> rough =
        Evaluate("hapke", hapke, cases, variance);
    const double tanT =
        std::tan(static_cast<double>(hapke[4]) * kPi / 180.0);
    hapke[4] = static_cast<float>(
        std::atan(std::sqrt(tanT * tanT + 2.0 * variance / kPi)) *
        180.0 / kPi);
    const std::vector<glm::dvec3> widened =
        Evaluate("hapke", hapke, cases);
    const std::vector<glm::dvec3> smooth = Evaluate(
        "hapke",
        BrdfRegistry::WithBuiltins().Find("hapke")->Defaults(),
        cases);
    double changed = 0.0;
    for (std::size_t c = 0; c < cases.size(); ++c) {
        for (int ch = 0; ch < 3; ++ch) {
            EXPECT_NEAR(rough[c][ch], widened[c][ch],
                        1e-4 * std::max(1.0, widened[c][ch]))
                << "case " << c;
            changed = std::max(
                changed, std::abs(rough[c][ch] - smooth[c][ch]));
        }
    }
    EXPECT_GT(changed, 1e-3) << "the variance makes a difference";

    const double unresolved =
        std::atan(std::sqrt(variance)) * 180.0 / kPi;
    const std::vector<glm::dvec3> orenNayar =
        Evaluate("oren_nayar", {0.0f}, cases, variance);
    const std::vector<glm::dvec3> orenNayarWidened = Evaluate(
        "oren_nayar", {static_cast<float>(unresolved)}, cases);
    for (std::size_t c = 0; c < cases.size(); ++c) {
        EXPECT_NEAR(orenNayar[c].r, orenNayarWidened[c].r, 1e-4)
            << "case " << c;
    }
    const std::vector<glm::dvec3> lambert =
        Evaluate("lambert", {}, cases, variance);
    for (std::size_t c = 0; c < cases.size(); ++c) {
        EXPECT_NEAR(lambert[c].r, cases[c].albedo.r * cases[c].mu0,
                    1e-5)
            << "Lambert ignores it";
    }
}

TEST_F(BrdfVulkanTest, PreviewSurfaceMatchesTheGltfReferenceBrdf) {
    // Dielectric, metal, the specular workflow and a clearcoat,
    // against the vector-form reference.
    const std::vector<std::vector<float>> parameterSets = {
        BrdfRegistry::WithBuiltins()
            .Find("preview_surface")
            ->Defaults(),
        {0.3f, 1.0f, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.01f},
        {0.8f, 0.4f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.01f},
        {0.5f, 0.0f, 1.5f, 1.0f, 0.2f, 0.5f, 0.8f, 0.0f, 0.01f},
        {0.6f, 0.0f, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.7f, 0.2f}};
    for (const auto& params : parameterSets) {
        ExpectMatches(
            "preview_surface", params,
            [&params](const Geometry& g, double a) {
                return PreviewSurfaceReference(g, a, params);
            },
            1e-3);
    }
    // Unresolved slopes widen the lobe as alpha^2 + 2 variance.
    const std::vector<Geometry> cases = GeometryGrid();
    const std::vector<float> params = {0.3f, 1.0f, 1.5f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 0.0f, 0.01f};
    const std::vector<glm::dvec3> rough =
        Evaluate("preview_surface", params, cases, 0.02);
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const double expected = PreviewSurfaceReference(
            cases[c], cases[c].albedo.g, params, 0.02);
        EXPECT_NEAR(rough[c].g, expected,
                    1e-3 * std::max(1.0, expected))
            << "case " << c;
    }
}

TEST_F(BrdfVulkanTest,
       HapkeHFunctionIsWithinOnePercentOfChandrasekhar) {
    // Isotropic, conservative scattering (w = 1) without opposition
    // effect, roughness or porosity reduces Hapke to I/F = H(mu0)
    // H(mu) / 8 at normal incidence and emission. Chandrasekhar's
    // exact H(1) for w = 1 is 2.90781.
    const std::vector<glm::dvec3> result =
        Evaluate("hapke", {0.0f, 0.0f, 0.0f, 0.07f, 0.0f, 0.0f},
                 {{1.0, 1.0, 1.0, 1.0, glm::dvec3(1.0)}});
    const double h = std::sqrt(8.0 * result[0].r);
    EXPECT_NEAR(h, 2.90781, 0.01 * 2.90781);
}

TEST_F(BrdfVulkanTest, SmoothModelsAreReciprocal) {
    // Without macroscopic roughness, the bidirectional reflectance
    // (I/F / mu0) is symmetric under exchanging incidence and
    // emission.
    std::vector<Geometry> forward = GeometryGrid();
    std::vector<Geometry> swapped = forward;
    for (Geometry& g : swapped) {
        std::swap(g.mu0, g.mu);
    }
    const std::vector<std::pair<std::string, std::vector<float>>>
        models = {
            {"lommel_seeliger", {0.2f}},
            {"hapke", {0.25f, 0.5f, 1.0f, 0.07f, 0.0f, 0.3f}},
            {"preview_surface",
             {0.4f, 0.5f, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.3f}}};
    for (const auto& [id, params] : models) {
        const std::vector<glm::dvec3> a =
            Evaluate(id, params, forward);
        const std::vector<glm::dvec3> b =
            Evaluate(id, params, swapped);
        for (std::size_t c = 0; c < forward.size(); ++c) {
            const double ra = a[c].g / forward[c].mu0;
            const double rb = b[c].g / swapped[c].mu0;
            EXPECT_NEAR(ra, rb, 1e-4 * std::max(1.0, ra))
                << id << " case " << c;
        }
    }
}

TEST_F(BrdfVulkanTest,
       LunarLambertWithConstantUnitWeightIsLommelSeeliger) {
    const std::vector<Geometry> cases = GeometryGrid();
    const auto lunar =
        Evaluate("lunar_lambert", {0.0f, 0.0f, 0.0f}, cases);
    const auto ls = Evaluate("lommel_seeliger", {0.0f}, cases);
    for (std::size_t c = 0; c < cases.size(); ++c) {
        EXPECT_NEAR(lunar[c].b, 8.0 * ls[c].b, 1e-5) << "case " << c;
    }
}

namespace {

/// Checks the sampling helpers for every case of (normal, light,
/// view): the mean `sampleBrdf` weight over a stratified grid, a
/// uniform-hemisphere quadrature of `evaluateBrdf` over the same
/// integral, the worst disagreement of a sample's pdf with `brdfPdf`
/// and of its direction and the basis with unit length and
/// orthogonality, and `brdfInputOf` of the case's light and view.
constexpr char kSamplingHarness[] = R"(import brdf;
import MODULE;

struct Case {
    float4 normal;
    float4 light;
    float4 view;
    float4 albedo;
}

[[vk::binding(0, 0)]] StructuredBuffer<Case> cases;
[[vk::binding(1, 0)]] StructuredBuffer<BrdfParams> params;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> results;

static const uint kGrid = 64;
static const uint kQuadrature = 256;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint count;
    uint stride;
    cases.GetDimensions(count, stride);
    if (id.x >= count) {
        return;
    }
    let c = cases[id.x];
    let n = c.normal.xyz;
    let v = c.view.xyz;
    let albedo = c.albedo.rgb;
    TYPE model;

    var sampled = float3(0.0);
    var pdfError = 0.0;
    var minCosine = 1.0;
    var lengthError = 0.0;
    for (uint i = 0; i < kGrid; ++i) {
        for (uint j = 0; j < kGrid; ++j) {
            let u = (float2(i, j) + 0.5) / float(kGrid);
            let s = sampleBrdf(model, n, v, albedo, params[0], u);
            sampled += s.weight;
            pdfError = max(pdfError, abs(s.pdf - brdfPdf(n, s.direction)));
            minCosine = min(minCosine, dot(n, s.direction));
            lengthError = max(lengthError, abs(length(s.direction) - 1.0));
        }
    }
    sampled /= float(kGrid * kGrid);

    float3 t;
    float3 b;
    orthonormalBasis(n, t, b);
    var quadrature = float3(0.0);
    for (uint i = 0; i < kQuadrature; ++i) {
        for (uint j = 0; j < kQuadrature; ++j) {
            let z = (float(i) + 0.5) / float(kQuadrature);
            let phi = 2.0 * PI * (float(j) + 0.5) / float(kQuadrature);
            let r = sqrt(max(0.0, 1.0 - z * z));
            let l = r * cos(phi) * t + r * sin(phi) * b + z * n;
            quadrature += evaluateBrdf(model, n, l, v, albedo, params[0]);
        }
    }
    quadrature *= 2.0 * PI / float(kQuadrature * kQuadrature);

    let basisError = abs(dot(t, b)) + abs(dot(t, n)) + abs(dot(b, n)) + abs(length(t) - 1.0) +
                     abs(length(b) - 1.0) + length(cross(t, b) - n);
    let input = brdfInputOf(n, c.light.xyz, v, albedo);
    let base = 5 * id.x;
    results[base] = float4(sampled, 0.0);
    results[base + 1] = float4(quadrature, 0.0);
    results[base + 2] = float4(pdfError, minCosine, lengthError, basisError);
    results[base + 3] = float4(input.mu0, input.mu, input.cosG, input.cosPsi);
    results[base + 4] = float4(evaluateBrdf(model, n, c.light.xyz, v, albedo, params[0]), 0.0);
}
)";

/// A unit normal, light and view direction.
struct SamplingCase {
    glm::dvec3 normal;
    glm::dvec3 light;
    glm::dvec3 view;
};

/// What `kSamplingHarness` returns for one case.
struct SamplingResult {
    glm::dvec3 sampled;
    glm::dvec3 quadrature;
    double pdfError;
    double minCosine;
    double lengthError;
    double basisError;
    glm::dvec4 input;
    glm::dvec3 evaluated;
};

/// Normals in every octant and both poles (the basis's sign switch),
/// with lights and views above, grazing and below the horizon.
std::vector<SamplingCase> SamplingCases() {
    const std::vector<glm::dvec3> normals = {{0.0, 0.0, 1.0},
                                             {0.0, 0.0, -1.0},
                                             {1.0, 0.0, 0.0},
                                             {0.3, -0.5, 0.8},
                                             {-0.6, 0.2, -0.7}};
    std::vector<SamplingCase> cases;
    for (const glm::dvec3& raw : normals) {
        const glm::dvec3 n = glm::normalize(raw);
        const glm::dvec3 helper = std::abs(n.x) < 0.9
                                      ? glm::dvec3(1.0, 0.0, 0.0)
                                      : glm::dvec3(0.0, 1.0, 0.0);
        const glm::dvec3 t = glm::normalize(glm::cross(helper, n));
        const glm::dvec3 b = glm::cross(n, t);
        const auto direction = [&](double zenith, double azimuth) {
            const double z = zenith * kPi / 180.0;
            const double a = azimuth * kPi / 180.0;
            return std::sin(z) * (std::cos(a) * t + std::sin(a) * b) +
                   std::cos(z) * n;
        };
        cases.push_back(
            {n, direction(30.0, 0.0), direction(0.0, 0.0)});
        cases.push_back(
            {n, direction(60.0, 20.0), direction(45.0, 200.0)});
        cases.push_back(
            {n, direction(75.0, 90.0), direction(70.0, 90.0)});
        cases.push_back(
            {n, direction(120.0, 0.0), direction(10.0, 0.0)});
    }
    return cases;
}

class BrdfSamplingTest : public BrdfVulkanTest {
  protected:
    std::vector<SamplingResult>
    Sample(const std::string& id, const std::vector<float>& values,
           const std::vector<SamplingCase>& cases,
           glm::dvec3 albedo) {
        const BrdfModel* model = registry_.Find(id);
        if (model == nullptr) {
            throw std::invalid_argument(id);
        }
        std::string harness = kSamplingHarness;
        harness.replace(harness.find("MODULE"), 6,
                        ernie::BrdfModuleName(*model));
        harness.replace(harness.find("TYPE"), 4, model->typeName);
        std::vector<ShaderModuleSource> modules =
            ernie::BrdfModules(*model);
        modules.push_back({"sampling_harness", harness});
        const ComputePipeline pipeline(
            *context_, compiler_.CompileSpirv(modules, "main"), 3);

        std::vector<float> packed;
        for (const SamplingCase& c : cases) {
            for (const glm::dvec3& v :
                 {c.normal, c.light, c.view, albedo}) {
                packed.insert(packed.end(),
                              {static_cast<float>(v.x),
                               static_cast<float>(v.y),
                               static_cast<float>(v.z), 0.0f});
            }
        }
        const ParamBuffers params = PackParams(*model, values);
        GpuBuffer caseBuffer(*context_, packed.size() * sizeof(float),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             BufferMemory::Device);
        const GpuBuffer& paramBuffer = *params.core;
        GpuBuffer resultBuffer(*context_,
                               cases.size() * 5 * 4 * sizeof(float),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               BufferMemory::Readback);
        caseBuffer.Write(std::as_bytes(std::span(packed)));
        const std::array<const GpuBuffer*, 3> buffers = {
            &caseBuffer, &paramBuffer, &resultBuffer};
        context_->Submit([&](VkCommandBuffer commands) {
            pipeline.Dispatch(
                commands, buffers,
                static_cast<std::uint32_t>((cases.size() + 63) / 64));
        });
        std::vector<float> raw(cases.size() * 5 * 4);
        resultBuffer.Read(std::as_writable_bytes(std::span(raw)));
        std::vector<SamplingResult> results;
        for (std::size_t c = 0; c < cases.size(); ++c) {
            const float* r = &raw[20 * c];
            results.push_back({{r[0], r[1], r[2]},
                               {r[4], r[5], r[6]},
                               r[8],
                               r[9],
                               r[10],
                               r[11],
                               {r[12], r[13], r[14], r[15]},
                               {r[16], r[17], r[18]}});
        }
        return results;
    }
};

}  // namespace

TEST_F(BrdfSamplingTest,
       LambertSampleWeightsAreTheAlbedoAndIntegrateToIt) {
    const glm::dvec3 albedo(0.12, 0.35, 0.8);
    const std::vector<SamplingCase> cases = SamplingCases();
    const std::vector<SamplingResult> results =
        Sample("lambert", {}, cases, albedo);
    for (std::size_t c = 0; c < cases.size(); ++c) {
        for (int ch = 0; ch < 3; ++ch) {
            EXPECT_NEAR(results[c].sampled[ch], albedo[ch], 1e-4)
                << "case " << c;
            EXPECT_NEAR(results[c].quadrature[ch], albedo[ch], 1e-3)
                << "case " << c;
        }
    }
}

TEST_F(BrdfSamplingTest, SamplesAreUnitAboveTheHorizonWithTheirPdf) {
    const std::vector<SamplingCase> cases = SamplingCases();
    const std::vector<SamplingResult> results =
        Sample("lommel_seeliger", {0.0f}, cases, glm::dvec3(0.5));
    for (std::size_t c = 0; c < cases.size(); ++c) {
        EXPECT_LT(results[c].pdfError, 1e-5) << "case " << c;
        EXPECT_GE(results[c].minCosine, 0.0) << "case " << c;
        EXPECT_LT(results[c].lengthError, 1e-5) << "case " << c;
        EXPECT_LT(results[c].basisError, 1e-5) << "case " << c;
    }
}

TEST_F(BrdfSamplingTest,
       ImportanceSamplingMatchesHemisphereQuadrature) {
    const std::vector<std::pair<std::string, std::vector<float>>>
        models = {
            {"lommel_seeliger", {-0.3f}},
            {"lunar_lambert", {-0.019f, 0.000242f, -0.00000146f}},
            {"oren_nayar", {25.0f}},
            {"hapke",
             BrdfRegistry::WithBuiltins().Find("hapke")->Defaults()}};
    const std::vector<SamplingCase> cases = SamplingCases();
    for (const auto& [id, params] : models) {
        const std::vector<SamplingResult> results =
            Sample(id, params, cases, glm::dvec3(0.12, 0.35, 0.8));
        for (std::size_t c = 0; c < cases.size(); ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                const double expected = results[c].quadrature[ch];
                EXPECT_GT(expected, 0.0) << id << " case " << c;
                EXPECT_NEAR(results[c].sampled[ch], expected,
                            0.01 * expected)
                    << id << " case " << c << " channel " << ch;
            }
        }
    }
}

TEST_F(BrdfSamplingTest, InputAndEvaluationFollowTheCaseGeometry) {
    const glm::dvec3 albedo(0.12, 0.35, 0.8);
    const std::vector<SamplingCase> cases = SamplingCases();
    const std::vector<SamplingResult> results =
        Sample("lambert", {}, cases, albedo);
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const SamplingCase& k = cases[c];
        const double mu0 = glm::dot(k.normal, k.light);
        const double mu = glm::dot(k.normal, k.view);
        const glm::dvec3 lPlane = k.light - k.normal * mu0;
        const glm::dvec3 vPlane = k.view - k.normal * mu;
        const double cosPsi =
            glm::length(lPlane) > 1e-5 && glm::length(vPlane) > 1e-5
                ? glm::dot(glm::normalize(lPlane),
                           glm::normalize(vPlane))
                : 1.0;
        EXPECT_NEAR(results[c].input.x, mu0, 1e-5) << "case " << c;
        EXPECT_NEAR(results[c].input.y, std::max(mu, 1e-3), 1e-5)
            << "case " << c;
        EXPECT_NEAR(results[c].input.z, glm::dot(k.light, k.view),
                    1e-5)
            << "case " << c;
        EXPECT_NEAR(results[c].input.w, cosPsi, 1e-5) << "case " << c;
        for (int ch = 0; ch < 3; ++ch) {
            EXPECT_NEAR(results[c].evaluated[ch],
                        albedo[ch] * std::max(mu0, 0.0) / kPi, 1e-6)
                << "case " << c;
        }
    }
}

namespace {

/// OpenPBR's defaults with `changes` (index, value) applied.
std::vector<float> OpenPbrParams(
    const BrdfModel& model,
    std::initializer_list<std::pair<std::size_t, float>> changes) {
    std::vector<float> values = model.Defaults();
    for (const auto& [index, value] : changes) {
        values[index] = value;
    }
    return values;
}

/// A view from `degrees` off the normal (0, 0, 1), and a light above.
SamplingCase ViewFrom(double degrees) {
    const double r = glm::radians(degrees);
    return {{0.0, 0.0, 1.0},
            {0.0, 0.0, 1.0},
            {std::sin(r), 0.0, std::cos(r)}};
}

}  // namespace

TEST_F(BrdfVulkanTest, OpenPbrWithoutSpecularIsLambertWhenSmooth) {
    // EON's multiple scattering vanishes at roughness 0, leaving
    // Lambert.
    const BrdfModel& model = *registry_.Find("open_pbr");
    ExpectMatches(
        "open_pbr", OpenPbrParams(model, {{3, 0.0f}}),
        [](const Geometry& g, double a) { return a * g.mu0; });
}

TEST_F(BrdfSamplingTest,
       OpenPbrRoughDiffuseKeepsAWhiteSurfacesEnergy) {
    // EON is energy preserving: a white rough base without specular
    // reflects all it receives, at any angle.
    const BrdfModel& model = *registry_.Find("open_pbr");
    for (const float roughness : {0.5f, 1.0f}) {
        const std::vector<SamplingResult> results =
            Sample("open_pbr",
                   OpenPbrParams(model, {{2, roughness}, {3, 0.0f}}),
                   {ViewFrom(0.0), ViewFrom(45.0), ViewFrom(75.0)},
                   glm::dvec3(1.0));
        for (const SamplingResult& result : results) {
            EXPECT_NEAR(result.sampled.x, 1.0, 0.02) << roughness;
        }
    }
}

TEST_F(BrdfVulkanTest,
       OpenPbrLayersMaterialsLackCostAndChangeNothing) {
    // Without a layer in a material (its weight 0) or in what the
    // model is compiled with (none of the scene's materials has it),
    // the result is the same as without the layer's code at all; a
    // layer compiled out is ignored even where a material has it.
    const BrdfModel& model = *registry_.Find("open_pbr");
    const std::vector<Geometry> cases = GeometryGrid();
    const std::vector<float> plain = model.Defaults();
    const std::vector<glm::dvec3> all =
        Evaluate("open_pbr", plain, cases);
    const std::vector<glm::dvec3> none =
        Evaluate("open_pbr", plain, cases, 0.0, "radiance", 0u);
    const std::vector<float> coated =
        OpenPbrParams(model, {{9, 1.0f}, {10, 1.0f}, {11, 1.0f}});
    const std::vector<glm::dvec3> compiledOut =
        Evaluate("open_pbr", coated, cases, 0.0, "radiance", 0u);
    const std::vector<glm::dvec3> layered =
        Evaluate("open_pbr", coated, cases);
    double changed = 0.0;
    for (std::size_t c = 0; c < cases.size(); ++c) {
        for (int ch = 0; ch < 3; ++ch) {
            EXPECT_NEAR(none[c][ch], all[c][ch], 1e-6) << c;
            EXPECT_NEAR(compiledOut[c][ch], all[c][ch], 1e-6) << c;
            changed = std::max(changed,
                               std::abs(layered[c][ch] - all[c][ch]));
        }
    }
    EXPECT_GT(changed, 0.01) << "the layers show where compiled in";
}

TEST_F(BrdfVulkanTest,
       OpenPbrThinFilmOfNoThicknessIsTheBareSpecular) {
    // A film of no thickness reflects as the substrate does; a thin
    // one's interference differs by wavelength, colouring it.
    const BrdfModel& model = *registry_.Find("open_pbr");
    const std::vector<Geometry> cases = GeometryGrid(glm::dvec3(0.5));
    const std::vector<glm::dvec3> bare = Evaluate(
        "open_pbr", model.Defaults(), cases, 0.0, "specular");
    const std::vector<glm::dvec3> none = Evaluate(
        "open_pbr", OpenPbrParams(model, {{11, 1.0f}, {22, 0.0f}}),
        cases, 0.0, "specular");
    const std::vector<glm::dvec3> film = Evaluate(
        "open_pbr", OpenPbrParams(model, {{11, 1.0f}, {22, 0.3f}}),
        cases, 0.0, "specular");
    double tinted = 0.0;
    for (std::size_t c = 0; c < cases.size(); ++c) {
        for (int ch = 0; ch < 3; ++ch) {
            EXPECT_NEAR(none[c][ch], bare[c][ch],
                        1e-4 * std::max(1.0, bare[c][ch]))
                << c;
        }
        if (bare[c].g > 1e-3) {
            tinted = std::max(
                tinted, std::abs(film[c].r - film[c].b) / film[c].g);
        }
    }
    EXPECT_GT(tinted, 0.1) << "interference colours the film";
}

TEST_F(BrdfSamplingTest,
       OpenPbrFuzzReflectsItsAlbedoAsItsEnvironmentSays) {
    // The sheen's distribution integrates to 1, so a fuzz over a
    // black base reflects its directional albedo, what `environment`
    // gives for light from all around.
    const BrdfModel& model = *registry_.Find("open_pbr");
    for (const float roughness : {0.3f, 0.7f}) {
        const std::vector<float> fuzz = OpenPbrParams(
            model, {{3, 0.0f}, {10, 1.0f}, {15, roughness}});
        for (const double degrees : {0.0, 45.0, 70.0}) {
            const SamplingResult direct =
                Sample("open_pbr", fuzz, {ViewFrom(degrees)},
                       glm::dvec3(0.0))[0];
            const double mu = std::cos(glm::radians(degrees));
            const glm::dvec3 environment =
                Evaluate("open_pbr", fuzz,
                         {{mu, mu, 1.0, 1.0, glm::dvec3(0.0)}}, 0.0,
                         "environment")[0];
            // Little head on, most towards grazing.
            EXPECT_GT(environment.x, 0.001);
            EXPECT_NEAR(direct.quadrature.x, environment.x,
                        0.05 * environment.x + 0.005)
                << roughness << " " << degrees;
        }
    }
}

TEST_F(BrdfVulkanTest,
       OpenPbrAnisotropyStretchesTheHighlightAlongTheTangent) {
    // A half vector tilted along the tangent, where the lobe is
    // stretched, is likelier than one tilted across it; without
    // anisotropy the surface's frame does not matter.
    const BrdfModel& model = *registry_.Find("open_pbr");
    const std::vector<Geometry> cases = {
        {std::cos(glm::radians(30.0)), std::cos(glm::radians(50.0)),
         std::cos(glm::radians(80.0)), -1.0, glm::dvec3(0.9)}};
    const std::vector<float> metal =
        OpenPbrParams(model, {{1, 1.0f}, {4, 0.3f}});
    const std::vector<float> brushed =
        OpenPbrParams(model, {{1, 1.0f}, {4, 0.3f}, {24, 0.8f}});
    const auto at = [&](const std::vector<float>& params,
                        double rotation) {
        return Evaluate("open_pbr", params, cases, 0.0, "radiance",
                        std::nullopt, rotation)[0]
            .g;
    };
    EXPECT_NEAR(at(metal, 0.0), at(metal, glm::half_pi<double>()),
                1e-4 * at(metal, 0.0));
    EXPECT_GT(at(brushed, 0.0),
              1.5 * at(brushed, glm::half_pi<double>()));
}

TEST_F(BrdfVulkanTest,
       OpenPbrMetalsEdgeTintShapesTheirEnvironmentToo) {
    // The F82 tint changes a metal's reflection of light from all
    // around towards grazing, as it does a light's, and hardly at all
    // head on, where F82's term vanishes.
    const BrdfModel& model = *registry_.Find("open_pbr");
    const std::vector<float> plain =
        OpenPbrParams(model, {{1, 1.0f}, {4, 0.2f}});
    const std::vector<float> tinted = OpenPbrParams(
        model,
        {{1, 1.0f}, {4, 0.2f}, {6, 0.5f}, {7, 0.5f}, {8, 0.5f}});
    const auto environment = [&](const std::vector<float>& params,
                                 double mu) {
        return Evaluate("open_pbr", params,
                        {{mu, mu, 1.0, 1.0, glm::dvec3(0.9)}}, 0.0,
                        "environment")[0]
            .g;
    };
    EXPECT_NEAR(environment(tinted, 1.0), environment(plain, 1.0),
                0.02 * environment(plain, 1.0));
    EXPECT_LT(environment(tinted, 0.1),
              0.95 * environment(plain, 0.1));
}

TEST_F(BrdfVulkanTest,
       OpenPbrSubsurfaceIsTheShareOfItsDiffuseItSpreads) {
    // Without specular, all the light of a base scattering wholly
    // beneath its surface is subsurface light, and half of it of one
    // scattering half; none without subsurface.
    const BrdfModel& model = *registry_.Find("open_pbr");
    const std::vector<Geometry> cases = GeometryGrid(glm::dvec3(0.5));
    const auto with = [&](float weight, const std::string& mode) {
        return Evaluate("open_pbr",
                        OpenPbrParams(model, {{3, 0.0f},
                                              {30, weight},
                                              {31, 0.5f},
                                              {32, 0.5f},
                                              {33, 0.5f}}),
                        cases, 0.0, mode);
    };
    const std::vector<glm::dvec3> whole = with(1.0f, "radiance");
    const std::vector<glm::dvec3> spread = with(1.0f, "subsurface");
    const std::vector<glm::dvec3> half = with(0.5f, "subsurface");
    const std::vector<glm::dvec3> none = with(0.0f, "subsurface");
    for (std::size_t c = 0; c < cases.size(); ++c) {
        EXPECT_NEAR(spread[c].g, whole[c].g,
                    1e-5 + 1e-4 * whole[c].g);
        EXPECT_NEAR(half[c].g, 0.5 * spread[c].g,
                    1e-5 + 1e-4 * spread[c].g);
        EXPECT_EQ(none[c].g, 0.0);
    }
}

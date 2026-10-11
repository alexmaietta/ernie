// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/shader_compiler.h"
#include "ernie/raster/atmosphere_luts.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

using ernie::AtmosphereLuts;
using ernie::AtmosphereParams;
using ernie::BandedAtmosphere;
using ernie::GpuImage;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

constexpr double kPi = std::numbers::pi;

/// Every layer of a float32 RGBA table, layer-major.
std::vector<float> TableOf(const ernie::VulkanContext& context,
                           const GpuImage& source) {
    auto& image = const_cast<GpuImage&>(source);  // only relaid out
    const ernie::ImageDesc& desc = image.Desc();
    const VkDeviceSize count =
        VkDeviceSize{desc.width} * desc.height * desc.layers * 4;
    ernie::GpuBuffer buffer(context, count * sizeof(float),
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            ernie::BufferMemory::Readback);
    const VkImageLayout layout = image.Layout();
    context.Submit([&](VkCommandBuffer commands) {
        image.Transition(commands,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
                                   desc.layers};
        region.imageExtent = {desc.width, desc.height, 1};
        vkCmdCopyImageToBuffer(commands, image.Handle(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               buffer.Handle(), 1, &region);
        image.Transition(commands, layout);
    });
    std::vector<float> values(count);
    buffer.Read(std::as_writable_bytes(std::span(values)));
    return values;
}

/// Texel `i`'s value of `size` texels in [0, 1], the ends on the
/// first and last centres (atmosphere.slang's `unitRange`).
double UnitOfTexel(int i, int size) {
    const double u = (i + 0.5) / size;
    return (u - 0.5 / size) / (1.0 - 1.0 / size);
}

/// Bruneton's inverse transmittance mapping, in metres.
void TransmittanceRMu(int x, int y, double bottom, double top,
                      double& r, double& mu) {
    const double xMu = UnitOfTexel(x, 256);
    const double xR = UnitOfTexel(y, 64);
    const double H = std::sqrt(top * top - bottom * bottom);
    const double rho = H * xR;
    r = std::sqrt(rho * rho + bottom * bottom);
    const double dMin = top - r;
    const double dMax = rho + H;
    const double d = dMin + xMu * (dMax - dMin);
    mu = d == 0.0
             ? 1.0
             : std::clamp((H * H - rho * rho - d * d) / (2.0 * r * d),
                          -1.0, 1.0);
}

/// Hillaire's Psi_ms at radius `r` (m) and sun cosine `muSun`, by the
/// estimator atmosphere_luts.slang uses (64 Fibonacci directions,
/// isotropic phase, a Lambertian ground) in double, with finer steps
/// and the reference transmittance in place of the table's.
double MultipleScatteringReference(const BandedAtmosphere& air,
                                   double r, double muSun, int band,
                                   double albedo) {
    const double bottom = air.bottomRadius;
    const double top = bottom + air.topAltitude;
    const auto scatteringAt = [&](double altitude) {
        double s = 0.0;
        for (const ernie::BandedConstituent& c : air.constituents) {
            s += static_cast<double>(
                     c.scattering[static_cast<std::size_t>(band)]) *
                 c.density(altitude);
        }
        return s;
    };
    const auto toSun = [&](double radius, double mu) {
        const double ground =
            radius * radius * (mu * mu - 1.0) + bottom * bottom;
        return mu < 0.0 && ground >= 0.0
                   ? 0.0
                   : air.TransmittanceToTop(radius, mu, band, 400);
    };
    constexpr int kDirections = 64;
    constexpr int kSteps = 200;
    const double sunX = std::sqrt(std::max(1.0 - muSun * muSun, 0.0));
    double second = 0.0;
    double transfer = 0.0;
    for (int i = 0; i < kDirections; ++i) {
        const double z = 1.0 - (2.0 * i + 1.0) / kDirections;
        const double radial = std::sqrt(std::max(1.0 - z * z, 0.0));
        const double phi = i * 2.39996322972865;
        const double dx = radial * std::cos(phi);
        const double dy = radial * std::sin(phi);
        const double groundDisc =
            r * r * (z * z - 1.0) + bottom * bottom;
        const bool ground = z < 0.0 && groundDisc >= 0.0;
        const double span =
            ground ? -r * z - std::sqrt(groundDisc)
                   : -r * z +
                         std::sqrt(r * r * (z * z - 1.0) + top * top);
        const double step = span / kSteps;
        double throughput = 1.0;
        for (int s = 0; s < kSteps; ++s) {
            const double t = (s + 0.5) * step;
            const double px = t * dx;
            const double py = t * dy;
            const double pz = r + t * z;
            const double radius =
                std::sqrt(px * px + py * py + pz * pz);
            const double altitude = radius - bottom;
            const double extinction = air.Extinction(altitude, band);
            const double scattering = scatteringAt(altitude);
            const double sample = std::exp(-extinction * step);
            const double fraction =
                extinction > 0.0 ? (1.0 - sample) / extinction : step;
            const double mu = (px * sunX + pz * muSun) / radius;
            second += throughput * scattering * toSun(radius, mu) /
                      (4.0 * kPi) * fraction / kDirections;
            transfer +=
                throughput * scattering * fraction / kDirections;
            throughput *= sample;
        }
        if (ground) {
            const double hx = span * dx;
            const double hy = span * dy;
            const double hz = r + span * z;
            const double length =
                std::sqrt(hx * hx + hy * hy + hz * hz);
            const double mu = (hx * sunX + hz * muSun) / length;
            second += throughput * toSun(bottom, mu) *
                      std::max(mu, 0.0) * albedo / kPi / kDirections;
        }
    }
    return second / (1.0 - transfer);
}

AtmosphereParams Earth() {
    AtmosphereParams params;
    params.preset = "earth";
    params.planetRadius = 6371.0088e3;
    return params;
}

}  // namespace

class AtmosphereLutsTest : public VulkanTest {};

// The transmittance table against the reference integral at its
// texels' own radii and angles, as optical depths: rays near the
// horizon cross over a thousand kilometres of air.
TEST_F(AtmosphereLutsTest, TransmittanceMatchesTheReference) {
    ernie::ShaderCompiler compiler;
    AtmosphereLuts luts(*context_, compiler);
    const ernie::SpectralBands bands{380.0, 780.0, 16};
    luts.Set(Earth(), bands);
    context_->Submit(
        [&](VkCommandBuffer commands) { luts.Record(commands); });
    const std::vector<float> table =
        TableOf(*context_, luts.Transmittance());
    const BandedAtmosphere& air = luts.Banded();
    const double bottom = air.bottomRadius;
    const double top = bottom + air.topAltitude;
    for (const int y : {0, 5, 31, 63}) {
        for (const int x : {0, 40, 128, 200, 250, 255}) {
            double r = 0.0;
            double mu = 0.0;
            TransmittanceRMu(x, y, bottom, top, r, mu);
            for (const int band : {0, 7, 15}) {
                const std::size_t texel =
                    (static_cast<std::size_t>(band / 4) * 64 * 256 +
                     static_cast<std::size_t>(y) * 256 +
                     static_cast<std::size_t>(x)) *
                        4 +
                    static_cast<std::size_t>(band % 4);
                const double depth = -std::log(table[texel]);
                const double reference =
                    -std::log(air.TransmittanceToTop(r, mu, band));
                EXPECT_NEAR(depth, reference, 3e-3 * reference + 1e-5)
                    << "texel " << x << ", " << y << " band " << band;
            }
        }
    }
}

TEST_F(AtmosphereLutsTest, MultipleScatteringMatchesTheReference) {
    ernie::ShaderCompiler compiler;
    AtmosphereLuts luts(*context_, compiler);
    const ernie::SpectralBands bands{380.0, 780.0, 16};
    luts.Set(Earth(), bands);
    context_->Submit(
        [&](VkCommandBuffer commands) { luts.Record(commands); });
    const std::vector<float> table =
        TableOf(*context_, luts.MultipleScattering());
    const BandedAtmosphere& air = luts.Banded();
    struct Texel {
        int x;
        int y;
    };
    // The sun overhead and low, at the ground and 10 km up.
    for (const Texel texel :
         {Texel{31, 0}, Texel{18, 0}, Texel{24, 3}}) {
        const double muSun = 2.0 * texel.x / 31.0 - 1.0;
        const double r =
            air.bottomRadius +
            std::max(air.topAltitude * texel.y / 31.0, 1.0);
        for (const int band : {1, 9}) {
            const std::size_t at =
                (static_cast<std::size_t>(band / 4) * 32 * 32 +
                 static_cast<std::size_t>(texel.y) * 32 +
                 static_cast<std::size_t>(texel.x)) *
                    4 +
                static_cast<std::size_t>(band % 4);
            const double reference =
                MultipleScatteringReference(air, r, muSun, band, 0.3);
            EXPECT_GT(reference, 0.0);
            EXPECT_NEAR(static_cast<double>(table[at]) / reference,
                        1.0, 0.03)
                << "texel " << texel.x << ", " << texel.y << " band "
                << band;
        }
    }
}

// The sun low over the ground: each band dims by the reference
// transmittance along its slant path, blue most.
TEST_F(RasterDelegateTest, TheAtmosphereDimsAndReddensTheSun) {
    constexpr double kElevation = 10.0;  // degrees
    const double e = kElevation * kPi / 180.0;
    const auto render = [&](bool atmosphere) {
        Scene scene = FlatScene();
        scene.Set("/World/Sun", tokens::kTransform,
                  ernie::DistantLightTransform(
                      {std::cos(e), 0.0, std::sin(e)}));
        if (atmosphere) {
            ernie::Atmosphere::Define(scene, "/World/Air");
        }
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        Tensor spectral = outputs.at("/Render/Product")
                              .vars.at("SpectralRadiance")
                              .Map(ernie::MapDevice::Cpu)
                              .tensor;
        const auto* values =
            static_cast<const float*>(spectral.Data());
        const std::size_t bandCount =
            static_cast<std::size_t>(spectral.Shape().front());
        const std::size_t c = kVarSize / 2;
        std::vector<double> centre;
        for (std::size_t b = 0; b < bandCount; ++b) {
            centre.push_back(
                values[(b * kVarSize + c) * kVarSize + c]);
        }
        std::optional<BandedAtmosphere> air;
        if (atmosphere) {
            air = delegate.Renderer().AtmosphereTables().Banded();
        }
        return std::make_pair(centre, air);
    };
    const auto [clear, none] = render(false);
    const auto [dimmed, air] = render(true);
    ASSERT_TRUE(air.has_value());
    ASSERT_EQ(clear.size(), dimmed.size());
    for (std::size_t b = 0; b < clear.size(); ++b) {
        ASSERT_GT(clear[b], 0.0);
        const double expected = air->TransmittanceToTop(
            air->bottomRadius, std::sin(e), static_cast<int>(b));
        EXPECT_NEAR(dimmed[b] / clear[b], expected, 0.01)
            << "band " << b;
    }
    EXPECT_LT(dimmed.front() / clear.front(),
              0.8 * dimmed.back() / clear.back())
        << "blue dims more than red";
}

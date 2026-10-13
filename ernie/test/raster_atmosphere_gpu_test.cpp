// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/shader_compiler.h"
#include "ernie/raster/atmosphere_luts.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <optional>
#include <tuple>
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

double Bilinear(const std::vector<float>& table, int width,
                int height, int band, double u, double v);

/// The sky irradiance table's value of band `band` with the sun at
/// zenith cosine `muSun`, at the fraction `height` of the atmosphere
/// (atmosphere.slang's `irradianceUv`).
double SkyIrradianceAt(const std::vector<float>& table, int band,
                       double muSun, double height) {
    const double x = (muSun + 0.2) / 1.2;
    const double u = (0.5 + x * 63.0) / 64.0;
    const double v = (0.5 + height * 15.0) / 16.0;
    return Bilinear(table, 64, 16, band, u, v);
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
        std::vector<float> sky;
        if (atmosphere) {
            const ernie::AtmosphereLuts& tables =
                delegate.Renderer().AtmosphereTables();
            air = tables.Banded();
            sky = TableOf(*context_, tables.SkyIrradiance());
        }
        return std::make_tuple(centre, air, sky);
    };
    const auto [clear, none, noSky] = render(false);
    const auto [dimmed, air, sky] = render(true);
    ASSERT_TRUE(air.has_value());
    ASSERT_EQ(clear.size(), dimmed.size());
    for (std::size_t b = 0; b < clear.size(); ++b) {
        ASSERT_GT(clear[b], 0.0);
        // The level ground under the sun dimmed by its slant path,
        // and lit by the sky: clear is albedo E cos / pi, so the sky
        // adds its irradiance over cos.
        const int band = static_cast<int>(b);
        const double toSun = air->TransmittanceToTop(
            air->bottomRadius, std::sin(e), band);
        const double fromSky =
            SkyIrradianceAt(sky, band, std::sin(e), 0.0) /
            std::sin(e);
        EXPECT_NEAR(dimmed[b] / clear[b], toSun + fromSky, 0.01)
            << "band " << b;
        EXPECT_GT(fromSky, 0.0) << "band " << b;
    }
}

namespace {

/// FlatScene without its terrain, under an Earth atmosphere, the
/// camera at `eye` looking at `target` and the sun `elevation`
/// degrees up in the +x direction.
Scene SkyScene(glm::dvec3 eye, glm::dvec3 target, double elevation) {
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    const double e = elevation * kPi / 180.0;
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform(
                  {std::cos(e), 0.0, std::sin(e)}));
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform(eye, target, {0.0, 1.0, 0.0}));
    ernie::Atmosphere::Define(scene, "/World/Air");
    return scene;
}

/// Every band's radiance at the pixel (row, col) of the
/// SpectralRadiance var.
std::vector<double>
SpectrumAt(ernie::RenderProductSetOutputs& outputs, std::size_t row,
           std::size_t col) {
    Tensor spectral = outputs.at("/Render/Product")
                          .vars.at("SpectralRadiance")
                          .Map(ernie::MapDevice::Cpu)
                          .tensor;
    const auto* values = static_cast<const float*>(spectral.Data());
    const auto bands =
        static_cast<std::size_t>(spectral.Shape().front());
    std::vector<double> spectrum;
    for (std::size_t b = 0; b < bands; ++b) {
        spectrum.push_back(
            values[(b * kVarSize + row) * kVarSize + col]);
    }
    return spectrum;
}

/// A table's value of band `band` at (u, v), bilinear between texel
/// centres and clamped at the edges, as a linear sampler reads it.
double Bilinear(const std::vector<float>& table, int width,
                int height, int band, double u, double v) {
    const double x = std::clamp(u * width - 0.5, 0.0, width - 1.0);
    const double y = std::clamp(v * height - 0.5, 0.0, height - 1.0);
    const int x0 = std::min(static_cast<int>(x), width - 2);
    const int y0 = std::min(static_cast<int>(y), height - 2);
    const auto at = [&](int i, int j) {
        const auto texel = static_cast<std::size_t>(
            ((band / 4) * height + j) * width + i);
        return static_cast<double>(
            table[texel * 4 + static_cast<std::size_t>(band % 4)]);
    };
    const double tx = x - x0;
    const double ty = y - y0;
    return (at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx) * (1 - ty) +
           (at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx) * ty;
}

/// The phase function of `constituent` in band `band` at cosine
/// `nu`, linear in u = sqrt(theta / pi) between its table's angles.
double PhaseAt(const BandedAtmosphere& air, std::size_t constituent,
               int band, double nu) {
    const auto& phase = air.constituents[constituent].phase;
    const int angles = air.phaseAngles;
    const double x =
        std::sqrt(std::acos(std::clamp(nu, -1.0, 1.0)) / kPi) *
        (angles - 1);
    const int i = std::min(static_cast<int>(x), angles - 2);
    const auto at = static_cast<std::size_t>(band * angles + i);
    return static_cast<double>(phase[at]) * (1.0 - (x - i)) +
           static_cast<double>(phase[at + 1]) * (x - i);
}

/// The light the air scatters along a vertical ray between the
/// altitudes `low` and `high` (the whole atmosphere when `high` is
/// negative), per unit sun irradiance, with the sun `elevation` up:
/// looking straight up from `low`, or `down` from `high`; and its
/// transmittance into `transmittance` when given. Single scattering
/// by each constituent's phase function and the reference
/// transmittance, multiple scattering from the renderer's Psi_ms
/// table, over 2000 steps.
double VerticalReference(const BandedAtmosphere& air,
                         const std::vector<float>& multiple, int band,
                         double elevation, bool down,
                         double low = 0.0, double high = -1.0,
                         double* through = nullptr) {
    const double muSun = std::sin(elevation);
    // The scattering cosine between the light's way and the view's.
    const double nu = down ? -muSun : muSun;
    const double bottom = air.bottomRadius;
    const double height = air.topAltitude;
    const double top = high < 0.0 ? height : high;
    constexpr int kSteps = 2000;
    const double step = (top - low) / kSteps;
    double transmittance = 1.0;
    double radiance = 0.0;
    for (int s = 0; s < kSteps; ++s) {
        const double altitude =
            down ? top - (s + 0.5) * step : low + (s + 0.5) * step;
        double extinction = 0.0;
        double scattering = 0.0;
        double phased = 0.0;
        for (std::size_t c = 0; c < air.constituents.size(); ++c) {
            const auto& constituent = air.constituents[c];
            const double density = constituent.density(altitude);
            const double sc =
                density *
                static_cast<double>(
                    constituent
                        .scattering[static_cast<std::size_t>(band)]);
            extinction +=
                sc + density *
                         static_cast<double>(
                             constituent
                                 .absorption[static_cast<std::size_t>(
                                     band)]);
            scattering += sc;
            phased += sc * PhaseAt(air, c, band, nu);
        }
        const double r = bottom + altitude;
        const double toSun =
            air.TransmittanceToTop(r, muSun, band, 400);
        const double u = (0.5 + (0.5 + 0.5 * muSun) * 31.0) / 32.0;
        const double v = (0.5 + altitude / height * 31.0) / 32.0;
        const double psi = Bilinear(multiple, 32, 32, band, u, v);
        const double source = phased * toSun + scattering * psi;
        const double sample = std::exp(-extinction * step);
        radiance += transmittance * source * (1.0 - sample) /
                    std::max(extinction, 1e-30);
        transmittance *= sample;
    }
    if (through != nullptr) {
        *through = transmittance;
    }
    return radiance;
}

}  // namespace

TEST_F(RasterDelegateTest, TheSkyMatchesTheReferenceAtTheZenith) {
    constexpr double kElevation = 30.0;
    const Scene scene =
        SkyScene({0.0, 0.0, 2.0}, {0.0, 0.0, 1000.0}, kElevation);
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const std::vector<double> sky =
        SpectrumAt(outputs, kVarSize / 2, kVarSize / 2);
    const ernie::AtmosphereLuts& tables =
        delegate.Renderer().AtmosphereTables();
    const BandedAtmosphere& air = tables.Banded();
    const std::vector<float> multiple =
        TableOf(*context_, tables.MultipleScattering());
    const std::vector<float> sun =
        ernie::SunIrradianceOf(scene, "/World/Sun", air.bands);
    ASSERT_EQ(sky.size(), sun.size());
    for (std::size_t b = 0; b < sky.size(); ++b) {
        const double reference =
            VerticalReference(air, multiple, static_cast<int>(b),
                              kElevation * kPi / 180.0, false) *
            static_cast<double>(sun[b]);
        EXPECT_NEAR(sky[b] / reference, 1.0, 0.03) << "band " << b;
    }
    // Per unit irradiance, blue skies: Rayleigh scattering's
    // lambda^-4 outweighs the haze.
    EXPECT_GT(sky.front() / static_cast<double>(sun.front()),
              2.0 * sky.back() / static_cast<double>(sun.back()));
}

// From orbit, space is black, and the air over the planet glows with
// the light it scatters up (no geometry stands for the planet's
// ground here: the atmosphere is the air alone).
TEST_F(RasterDelegateTest, FromOrbitSpaceIsBlackAndTheAirGlows) {
    constexpr double kElevation = 30.0;
    const double altitude = 1000e3;
    // Straight down at the planet, and up into space.
    const std::vector<double> down = [&] {
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs =
            delegate.Step(SkyScene({0.0, 0.0, altitude},
                                   {0.0, 0.0, 0.0}, kElevation));
        return SpectrumAt(outputs, kVarSize / 2, kVarSize / 2);
    }();
    RasterDelegate delegate(*context_);
    const Scene up = SkyScene({0.0, 0.0, altitude},
                              {0.0, 0.0, 2.0 * altitude}, kElevation);
    ernie::RenderProductSetOutputs outputs = delegate.Step(up);
    const std::vector<double> space =
        SpectrumAt(outputs, kVarSize / 2, kVarSize / 2);
    const ernie::AtmosphereLuts& tables =
        delegate.Renderer().AtmosphereTables();
    const BandedAtmosphere& air = tables.Banded();
    const std::vector<float> multiple =
        TableOf(*context_, tables.MultipleScattering());
    const std::vector<float> sun =
        ernie::SunIrradianceOf(up, "/World/Sun", air.bands);
    const double e = kElevation * kPi / 180.0;
    for (std::size_t b = 0; b < down.size(); ++b) {
        EXPECT_EQ(space[b], 0.0) << "band " << b;
        const int band = static_cast<int>(b);
        const double scattered =
            VerticalReference(air, multiple, band, e, true);
        EXPECT_NEAR(down[b] /
                        (static_cast<double>(sun[b]) * scattered),
                    1.0, 0.03)
            << "band " << b;
    }
}

// The sun's limb-darkened disc carries the sun's irradiance through
// the air: summed over its pixels (less the sky around it) it is the
// irradiance times the transmittance along the line of sight.
TEST_F(RasterDelegateTest, TheSunsDiscCarriesItsDimmedIrradiance) {
    constexpr double kElevation = 30.0;
    const double e = kElevation * kPi / 180.0;
    const glm::dvec3 towards(std::cos(e), 0.0, std::sin(e));
    Scene scene =
        SkyScene({0.0, 0.0, 2.0}, glm::dvec3(0.0, 0.0, 2.0) + towards,
                 kElevation);
    scene.Set("/World/Air", tokens::kAtmosphereHazeOpticalDepth, 0.0);
    // 1 mm of sensor at 50 mm: 1.15 degrees across, so the 0.53
    // degree disc spans some 15 pixels.
    ernie::Camera camera(scene, "/World/Camera");
    camera.SetFocalLength(50.0)
        .SetHorizontalAperture(1.0)
        .SetVerticalAperture(1.0);
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const BandedAtmosphere& air =
        delegate.Renderer().AtmosphereTables().Banded();
    const std::vector<float> sun =
        ernie::SunIrradianceOf(scene, "/World/Sun", air.bands);
    const double pixel = 1.0 / kVarSize / 50.0;  // radians
    for (const std::size_t band : {std::size_t{2}, std::size_t{12}}) {
        // The sky around the disc, from the image's corners.
        const std::size_t last = kVarSize - 1;
        double background = 0.0;
        for (const std::size_t row : {std::size_t{0}, last}) {
            for (const std::size_t col : {std::size_t{0}, last}) {
                background +=
                    SpectrumAt(outputs, row, col)[band] / 4.0;
            }
        }
        double irradiance = 0.0;
        for (std::uint32_t row = 0; row < kVarSize; ++row) {
            for (std::uint32_t col = 0; col < kVarSize; ++col) {
                const double excess =
                    SpectrumAt(outputs, row, col)[band] - background;
                const double x = (col + 0.5) - kVarSize / 2.0;
                const double y = (row + 0.5) - kVarSize / 2.0;
                if (x * x + y * y < 10.0 * 10.0) {
                    irradiance += excess * pixel * pixel;
                }
            }
        }
        const double expected =
            static_cast<double>(sun[band]) *
            air.TransmittanceToTop(air.bottomRadius + 2.0,
                                   std::sin(e),
                                   static_cast<int>(band));
        EXPECT_NEAR(irradiance / expected, 1.0, 0.03)
            << "band " << band;
    }
}

// Aerial perspective: the ground 2 km below the camera, seen through
// the air, dims by its transmittance and gains the sun's light the
// air scatters up on the way.
TEST_F(RasterDelegateTest, TheAirHazesTheGroundBelow) {
    constexpr double kElevation = 30.0;
    const double e = kElevation * kPi / 180.0;
    const double height = 2000.0;
    const auto render = [&](bool atmosphere) {
        Scene scene = FlatScene();
        scene.Set("/World/Sun", tokens::kTransform,
                  ernie::DistantLightTransform(
                      {std::cos(e), 0.0, std::sin(e)}));
        scene.Set("/World/Camera", tokens::kTransform,
                  ernie::LookAtTransform({0.0, 0.0, height},
                                         {0.0, 0.0, 0.0},
                                         {0.0, 1.0, 0.0}));
        if (atmosphere) {
            ernie::Atmosphere::Define(scene, "/World/Air");
        }
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        std::vector<double> spectrum =
            SpectrumAt(outputs, kVarSize / 2, kVarSize / 2);
        std::optional<BandedAtmosphere> air;
        std::vector<float> multiple;
        std::vector<float> sky;
        if (atmosphere) {
            const ernie::AtmosphereLuts& tables =
                delegate.Renderer().AtmosphereTables();
            air = tables.Banded();
            multiple =
                TableOf(*context_, tables.MultipleScattering());
            sky = TableOf(*context_, tables.SkyIrradiance());
        }
        return std::make_tuple(
            spectrum, air, multiple, sky,
            ernie::SunIrradianceOf(scene, "/World/Sun",
                                   air ? air->bands
                                       : ernie::SpectralBands{}));
    };
    const auto [clear, none, noTable, noSky, noSun] = render(false);
    const auto [hazed, air, multiple, sky, sun] = render(true);
    ASSERT_TRUE(air.has_value());
    for (std::size_t b = 0; b < clear.size(); ++b) {
        const int band = static_cast<int>(b);
        // The ground itself under the dimmed sun and the sky (clear
        // is albedo E cos / pi), then the air between.
        const double toSun = air->TransmittanceToTop(
            air->bottomRadius, std::sin(e), band);
        const double fromSky =
            SkyIrradianceAt(sky, band, std::sin(e), 0.0) /
            std::sin(e);
        double through = 1.0;
        const double scattered = VerticalReference(
            *air, multiple, band, e, true, 0.0, height, &through);
        const double expected =
            clear[b] * (toSun + fromSky) * through +
            static_cast<double>(sun[b]) * scattered;
        EXPECT_NEAR(hazed[b] / expected, 1.0, 0.02) << "band " << b;
    }
}

namespace {

/// The sky's radiance per unit sun irradiance seen from radius `r`
/// on the z axis along unit `d` (upwards), the sun along unit `sun`:
/// single scattering by each constituent's phase function and the
/// reference transmittance, multiple scattering from the renderer's
/// Psi_ms table, over 400 steps to the top.
double SkyRadianceReference(const BandedAtmosphere& air,
                            const std::vector<float>& multiple,
                            int band, double r, glm::dvec3 d,
                            glm::dvec3 sun) {
    const double bottom = air.bottomRadius;
    const double top = bottom + air.topAltitude;
    const double length =
        -r * d.z + std::sqrt(r * r * (d.z * d.z - 1.0) + top * top);
    constexpr int kSteps = 400;
    const double step = length / kSteps;
    const double nu = glm::dot(d, sun);
    const auto b = static_cast<std::size_t>(band);
    double transmittance = 1.0;
    double radiance = 0.0;
    for (int s = 0; s < kSteps; ++s) {
        const glm::dvec3 point =
            glm::dvec3(0.0, 0.0, r) + (s + 0.5) * step * d;
        const double radius = glm::length(point);
        const double altitude = radius - bottom;
        double extinction = 0.0;
        double scattering = 0.0;
        double phased = 0.0;
        for (std::size_t c = 0; c < air.constituents.size(); ++c) {
            const auto& constituent = air.constituents[c];
            const double density = constituent.density(altitude);
            const double sc =
                density *
                static_cast<double>(constituent.scattering[b]);
            extinction +=
                sc + density * static_cast<double>(
                                   constituent.absorption[b]);
            scattering += sc;
            phased += sc * PhaseAt(air, c, band, nu);
        }
        const double muSun = glm::dot(point, sun) / radius;
        const double ground =
            radius * radius * (muSun * muSun - 1.0) + bottom * bottom;
        const double toSun =
            muSun < 0.0 && ground >= 0.0
                ? 0.0
                : air.TransmittanceToTop(radius, muSun, band, 200);
        const double u = (0.5 + (0.5 + 0.5 * muSun) * 31.0) / 32.0;
        const double v =
            (0.5 + std::clamp(altitude / air.topAltitude, 0.0, 1.0) *
                       31.0) /
            32.0;
        const double psi = Bilinear(multiple, 32, 32, band, u, v);
        const double sample = std::exp(-extinction * step);
        radiance += transmittance *
                    (phased * toSun + scattering * psi) *
                    (1.0 - sample) / std::max(extinction, 1e-30);
        transmittance *= sample;
    }
    return radiance;
}

}  // namespace

// The sky irradiance table against the reference sky's radiance
// integrated over the hemisphere of level ground by 256 directions
// (cosine weighted on a Fibonacci spiral), four times the table's.
TEST_F(AtmosphereLutsTest, SkyIrradianceMatchesTheReference) {
    ernie::ShaderCompiler compiler;
    AtmosphereLuts luts(*context_, compiler);
    luts.Set(Earth(), {380.0, 780.0, 16});
    context_->Submit(
        [&](VkCommandBuffer commands) { luts.Record(commands); });
    const std::vector<float> multiple =
        TableOf(*context_, luts.MultipleScattering());
    const std::vector<float> table =
        TableOf(*context_, luts.SkyIrradiance());
    const BandedAtmosphere& air = luts.Banded();
    // The sun overhead and 10 degrees up, at the ground (1 m up, as
    // the table's rays start).
    for (const int x : {63, 21}) {
        const double muSun = -0.2 + 1.2 * x / 63.0;
        const glm::dvec3 sun(std::sqrt(1.0 - muSun * muSun), 0.0,
                             muSun);
        for (const int band : {1, 12}) {
            constexpr int kDirections = 256;
            double irradiance = 0.0;
            for (int i = 0; i < kDirections; ++i) {
                const double disc =
                    std::sqrt((i + 0.5) / kDirections);
                const double phi = i * 2.39996322972865;
                const glm::dvec3 d(disc * std::cos(phi),
                                   disc * std::sin(phi),
                                   std::sqrt(1.0 - disc * disc));
                irradiance += SkyRadianceReference(
                    air, multiple, band, air.bottomRadius + 1.0, d,
                    sun);
            }
            irradiance *= kPi / kDirections;
            const std::size_t at =
                (static_cast<std::size_t>(band / 4) * 16 * 64 +
                 static_cast<std::size_t>(x)) *
                    4 +
                static_cast<std::size_t>(band % 4);
            EXPECT_NEAR(static_cast<double>(table[at]) / irradiance,
                        1.0, 0.03)
                << "sun cosine " << muSun << " band " << band;
        }
    }
}

// The refraction table against the CPU's ray trace at its texels'
// own heights and apparent zenith angles, from overhead to the
// horizon, at the ground, partway up and at the top (where rays below
// the horizon cross the atmosphere and leave), in a blue and a red
// band, for rays that leave the atmosphere.
TEST_F(AtmosphereLutsTest, RefractionMatchesTheReference) {
    ernie::ShaderCompiler compiler;
    AtmosphereLuts luts(*context_, compiler);
    luts.Set(Earth(), {380.0, 780.0, 16});
    context_->Submit(
        [&](VkCommandBuffer commands) { luts.Record(commands); });
    const std::vector<float> table =
        TableOf(*context_, luts.Refraction());
    const BandedAtmosphere& air = luts.Banded();
    const double bottom = air.bottomRadius;
    const double height = air.topAltitude;
    int compared = 0;
    for (const int y : {0, 8, 31}) {
        for (const int x : {0, 128, 220, 245, 252}) {
            // refractionRMu: height up by the square, zenith angle
            // across crowded towards 2 degrees past the horizon.
            const double up = UnitOfTexel(y, 32);
            const double r =
                std::clamp(bottom + up * up * height, bottom + 0.1,
                           bottom + height - 0.1);
            const double horizon =
                std::acos(-std::sqrt(std::max(
                    1.0 - (bottom / r) * (bottom / r), 0.0))) +
                0.0349;
            const double s = 1.0 - UnitOfTexel(x, 256);
            const double mu = std::cos(horizon * (1.0 - s * s));
            for (const int band : {1, 14}) {
                const BandedAtmosphere::Refraction reference =
                    air.Refract(r, mu, band, 50.0);
                if (reference.ground) {
                    continue;
                }
                const std::size_t at =
                    (static_cast<std::size_t>(band / 4) * 32 * 256 +
                     static_cast<std::size_t>(y) * 256 +
                     static_cast<std::size_t>(x)) *
                        4 +
                    static_cast<std::size_t>(band % 4);
                EXPECT_NEAR(static_cast<double>(table[at]),
                            reference.deviation,
                            0.02 * reference.deviation + 2e-6)
                    << "texel " << x << ", " << y << " band " << band;
                ++compared;
            }
        }
    }
    EXPECT_GE(compared, 10);
}

// Refraction at sunset: the sun's true centre on the geometric
// horizon shows whole, lifted by about half a degree, and flattened,
// as the bending grows towards the horizon (its lower limb rises more
// than its upper).
TEST_F(RasterDelegateTest, RefractionLiftsAndFlattensTheSettingSun) {
    // 1 mm of sensor at 50 mm: 1.15 degrees across, the 0.53 degree
    // disc some 15 pixels wide; aimed half a degree up.
    Scene scene =
        SkyScene({0.0, 0.0, 2.0}, {1000.0, 0.0, 2.0 + 8.73}, 0.0);
    scene.Set("/World/Air", tokens::kAtmosphereHazeOpticalDepth, 0.0);
    ernie::Camera(scene, "/World/Camera")
        .SetFocalLength(50.0)
        .SetHorizontalAperture(1.0)
        .SetVerticalAperture(1.0);
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    // The disc: pixels brighter than half its brightest in a red
    // band.
    constexpr std::size_t kBand = 12;
    double brightest = 0.0;
    for (std::size_t row = 0; row < kVarSize; ++row) {
        for (std::size_t col = 0; col < kVarSize; ++col) {
            brightest = std::max(
                brightest, SpectrumAt(outputs, row, col)[kBand]);
        }
    }
    std::size_t top = kVarSize;
    std::size_t bottom = 0;
    std::size_t left = kVarSize;
    std::size_t right = 0;
    for (std::size_t row = 0; row < kVarSize; ++row) {
        for (std::size_t col = 0; col < kVarSize; ++col) {
            if (SpectrumAt(outputs, row, col)[kBand] >
                0.5 * brightest) {
                top = std::min(top, row);
                bottom = std::max(bottom, row);
                left = std::min(left, col);
                right = std::max(right, col);
            }
        }
    }
    ASSERT_LE(top, bottom);
    // SkyScene's camera is +y up while it looks along +x, so the
    // horizon runs down the image: rows run across it, columns up.
    const double across = static_cast<double>(bottom - top + 1);
    const double up = static_cast<double>(right - left + 1);
    EXPECT_GT(across, 12.0) << "the whole disc shows";
    EXPECT_LT(up / across, 0.92) << "flattened";
    // Lifted: the camera aims half a degree up and the disc's centre
    // shows near the middle rather than 14 pixels below it, where
    // its true centre, on the horizon, lies.
    const double centre = 0.5 * static_cast<double>(left + right);
    EXPECT_LT(std::abs(centre - (kVarSize / 2.0 - 0.5)), 5.0);
}

namespace {

/// The fraction of the sun's limb-darkened light (`sunDisc`'s power
/// law at `wavelength` nm) left by a disc of angular radius
/// `bodyRadius` whose centre is `separation` from the sun's, of
/// angular radius `sunRadius`: integrated over a fine polar grid of
/// the sun's disc.
double EclipseReference(double sunRadius, double bodyRadius,
                        double separation, double wavelength) {
    const double alpha = std::max(-0.023 + 292.0 / wavelength, 0.0);
    constexpr int kRings = 800;
    constexpr int kSectors = 800;
    double total = 0.0;
    double visible = 0.0;
    for (int i = 0; i < kRings; ++i) {
        const double rho = (i + 0.5) / kRings;
        const double weight =
            std::pow(std::sqrt(1.0 - rho * rho), alpha) * rho;
        for (int j = 0; j < kSectors; ++j) {
            const double phi = (j + 0.5) / kSectors * 2.0 * kPi;
            const double x =
                rho * sunRadius * std::cos(phi) - separation;
            const double y = rho * sunRadius * std::sin(phi);
            total += weight;
            if (x * x + y * y >= bodyRadius * bodyRadius) {
                visible += weight;
            }
        }
    }
    return visible / total;
}

constexpr double kMoonDistance = 384400e3;
constexpr double kMoonRadius = 1737.4e3;

/// A CelestialBody of the Moon's radius at its distance from the
/// world's origin, `offset` radians from `towards` (a unit vector)
/// within the x-z plane.
void MoonNear(Scene& scene, glm::dvec3 towards, double offset) {
    const glm::dvec3 across =
        glm::normalize(glm::cross(towards, {0.0, 1.0, 0.0}));
    const glm::dvec3 direction =
        std::cos(offset) * towards + std::sin(offset) * across;
    ernie::CelestialBody::Define(scene, "/World/Moon");
    scene.Set(
        "/World/Moon", tokens::kTransform,
        glm::translate(glm::dmat4(1.0), kMoonDistance * direction));
}

}  // namespace

// The Moon partly before the sun dims the ground below by the share
// of the limb-darkened disc it covers, which differs with the
// wavelength as the limb darkening does; wholly before it, to
// nothing.
TEST_F(RasterDelegateTest, TheMoonEclipsesTheGround) {
    constexpr double kElevation = 60.0;
    const double e = kElevation * kPi / 180.0;
    const glm::dvec3 towards(std::cos(e), 0.0, std::sin(e));
    const double sunRadius = 0.5 * 0.53 * kPi / 180.0;
    const double moonRadius = std::asin(kMoonRadius / kMoonDistance);
    const auto centre = [&](std::optional<double> offset) {
        Scene scene = FlatScene();
        scene.Set("/World/Sun", tokens::kTransform,
                  ernie::DistantLightTransform(towards));
        if (offset) {
            MoonNear(scene, towards, *offset);
        }
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        return SpectrumAt(outputs, kVarSize / 2, kVarSize / 2);
    };
    const std::vector<double> clear = centre(std::nullopt);
    const ernie::SpectralBands bands;
    for (const double offset : {0.6 * sunRadius, 1.5 * sunRadius}) {
        const std::vector<double> eclipsed = centre(offset);
        ASSERT_EQ(eclipsed.size(), clear.size());
        for (const int band : {1, 14}) {
            const auto b = static_cast<std::size_t>(band);
            const double expected = EclipseReference(
                sunRadius, moonRadius, offset, bands.Centre(band));
            EXPECT_NEAR(eclipsed[b] / clear[b], expected, 0.01)
                << "band " << band << " offset " << offset;
        }
        // Blue light gathers more towards the disc's centre: covering
        // it takes more blue, covering the limb more red.
        const bool central = offset < sunRadius;
        EXPECT_EQ(eclipsed[1] / clear[1] < eclipsed[14] / clear[14],
                  central)
            << "offset " << offset;
    }
    // The Moon three times as big covers the whole sun.
    Scene scene = FlatScene();
    scene.Set("/World/Sun", tokens::kTransform,
              ernie::DistantLightTransform(towards));
    MoonNear(scene, towards, 0.0);
    scene.Set("/World/Moon", tokens::kBodyRadius, 3.0 * kMoonRadius);
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    for (const double value :
         SpectrumAt(outputs, kVarSize / 2, kVarSize / 2)) {
        EXPECT_EQ(value, 0.0);
    }
}

// The same through the sun's long-range shadow view: a Moon of real
// geometry, a Sphere, eclipses the ground by the share of the
// limb-darkened disc its silhouette covers, as the analytic eclipse
// does; three times as big, wholly.
TEST_F(RasterDelegateTest, ASphereMoonEclipsesTheGround) {
    constexpr double kElevation = 60.0;
    const double e = kElevation * kPi / 180.0;
    const glm::dvec3 towards(std::cos(e), 0.0, std::sin(e));
    const double sunRadius = 0.5 * 0.53 * kPi / 180.0;
    const double moonRadius = std::asin(kMoonRadius / kMoonDistance);
    const glm::dvec3 across =
        glm::normalize(glm::cross(towards, {0.0, 1.0, 0.0}));
    const auto centre = [&](std::optional<double> offset,
                            double radius = kMoonRadius) {
        Scene scene = FlatScene();
        ernie::DistantLight(scene, "/World/Sun")
            .SetShadowEnable(true);
        scene.Set("/World/Sun", tokens::kTransform,
                  ernie::DistantLightTransform(towards));
        if (offset) {
            ernie::Sphere::Define(scene, "/World/Moon")
                .SetRadius(radius);
            scene.Set(
                "/World/Moon", tokens::kTransform,
                glm::translate(glm::dmat4(1.0),
                               kMoonDistance *
                                   (std::cos(*offset) * towards +
                                    std::sin(*offset) * across)));
        }
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        return SpectrumAt(outputs, kVarSize / 2, kVarSize / 2);
    };
    const std::vector<double> clear = centre(std::nullopt);
    const ernie::SpectralBands bands;
    for (const double offset : {0.6 * sunRadius, 1.5 * sunRadius}) {
        const std::vector<double> eclipsed = centre(offset);
        ASSERT_EQ(eclipsed.size(), clear.size());
        for (const int band : {1, 14}) {
            const auto b = static_cast<std::size_t>(band);
            const double expected = EclipseReference(
                sunRadius, moonRadius, offset, bands.Centre(band));
            EXPECT_NEAR(eclipsed[b] / clear[b], expected, 0.01)
                << "band " << band << " offset " << offset;
        }
        const bool central = offset < sunRadius;
        EXPECT_EQ(eclipsed[1] / clear[1] < eclipsed[14] / clear[14],
                  central)
            << "offset " << offset;
    }
    for (const double value : centre(0.0, 3.0 * kMoonRadius)) {
        EXPECT_NEAR(value, 0.0, 1e-6);
    }
}

// A Sphere Moon at its farthest, centred on the sun, leaves its ring:
// the ground keeps the limb-darkened annulus's share of the light.
TEST_F(RasterDelegateTest, ASphereMoonsAnnularEclipseLeavesTheRing) {
    constexpr double kFarthest = 405500e3;
    const glm::dvec3 towards =
        glm::normalize(glm::dvec3(0.5, 0.0, 0.866));
    const double sunRadius = 0.5 * 0.53 * kPi / 180.0;
    const auto centre = [&](bool moon) {
        Scene scene = FlatScene();
        ernie::DistantLight(scene, "/World/Sun")
            .SetShadowEnable(true);
        scene.Set("/World/Sun", tokens::kTransform,
                  ernie::DistantLightTransform(towards));
        if (moon) {
            ernie::Sphere::Define(scene, "/World/Moon")
                .SetRadius(kMoonRadius);
            scene.Set(
                "/World/Moon", tokens::kTransform,
                glm::translate(glm::dmat4(1.0), kFarthest * towards));
        }
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        return SpectrumAt(outputs, kVarSize / 2, kVarSize / 2);
    };
    const std::vector<double> clear = centre(false);
    const std::vector<double> ring = centre(true);
    const ernie::SpectralBands bands;
    for (const int band : {1, 14}) {
        const auto b = static_cast<std::size_t>(band);
        EXPECT_NEAR(ring[b] / clear[b],
                    EclipseReference(
                        sunRadius, std::asin(kMoonRadius / kFarthest),
                        0.0, bands.Centre(band)),
                    0.01)
            << "band " << band;
    }
}

// Phobos, a body of any shape that casts no analytic eclipse, crosses
// the smaller Martian sun: under it the ground keeps the share of the
// limb-darkened disc it leaves, the Martian air and its planet aside
// (the planet's shadow its atmosphere's own).
TEST_F(RasterDelegateTest, PhobosTransitsTheSunOverMars) {
    constexpr double kPhobosRadius = 11.27e3;
    constexpr double kPhobosAltitude = 6000e3;
    constexpr double kMarsSunAngle = 0.35;
    const glm::dvec3 towards =
        glm::normalize(glm::dvec3(0.5, 0.0, 0.866));
    const auto centre = [&](bool phobos) {
        Scene scene = FlatScene();
        ernie::Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kMars));
        ernie::DistantLight(scene, "/World/Sun")
            .SetShadowEnable(true)
            .SetAngle(kMarsSunAngle);
        scene.Set("/World/Sun", tokens::kTransform,
                  ernie::DistantLightTransform(towards));
        if (phobos) {
            ernie::Sphere::Define(scene, "/World/Phobos")
                .SetRadius(kPhobosRadius);
            scene.Set("/World/Phobos", tokens::kTransform,
                      glm::translate(glm::dmat4(1.0),
                                     kPhobosAltitude / towards.z *
                                         towards));
        }
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        return SpectrumAt(outputs, kVarSize / 2, kVarSize / 2);
    };
    const std::vector<double> clear = centre(false);
    const std::vector<double> transit = centre(true);
    const ernie::SpectralBands bands;
    const double sunRadius = 0.5 * kMarsSunAngle * kPi / 180.0;
    const double phobosRadius =
        std::asin(kPhobosRadius * towards.z / kPhobosAltitude);
    for (const int band : {1, 14}) {
        const auto b = static_cast<std::size_t>(band);
        EXPECT_NEAR(transit[b] / clear[b],
                    EclipseReference(sunRadius, phobosRadius, 0.0,
                                     bands.Centre(band)),
                    0.02)
            << "band " << band;
    }
}

// The Moon's umbra seen from geostationary orbit on a planet with an
// atmosphere: a Sphere Moon's through the long-range view as dark as
// the analytic eclipse's CelestialBody in it, and as bright in its
// penumbra.
TEST_F(RasterDelegateTest,
       TheUmbraFromOrbitMatchesTheAnalyticEclipse) {
    constexpr double kEarth = 6371008.8;
    const glm::dvec3 towards =
        glm::normalize(glm::dvec3(0.5, 0.0, 0.866));
    const double radius = 1.1 * kMoonRadius;
    enum class Moon { None, Body, Sphere };
    const auto view = [&](Moon moon) {
        Scene scene = FlatScene();
        scene.Remove("/World/Terrain");
        ernie::Terrain::Define(scene, "/World/Earth")
            .SetHeights(Tensor::Copy<float>(
                std::vector<float>(6 * 65 * 65, 0.0f), {6, 65, 65}))
            .SetPlanetRadius(kEarth)
            .SetMaterialBinding("/Looks/Grey");
        scene.Set("/World/Earth", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0),
                                 glm::dvec3(0.0, 0.0, -kEarth)));
        ernie::Atmosphere::Define(scene, "/World/Air");
        ernie::DistantLight(scene, "/World/Sun")
            .SetShadowEnable(true);
        scene.Set("/World/Sun", tokens::kTransform,
                  ernie::DistantLightTransform(towards));
        ernie::Camera(scene, "/World/Camera")
            .SetClippingRange(glm::dvec2(1e3, 1e9));
        scene.Set("/World/Camera", tokens::kTransform,
                  ernie::LookAtTransform({0.0, 0.0, 35786e3},
                                         {0.0, 0.0, 0.0},
                                         {0.0, 1.0, 0.0}));
        const glm::dmat4 at =
            glm::translate(glm::dmat4(1.0), 360000e3 * towards);
        if (moon == Moon::Body) {
            ernie::CelestialBody::Define(scene, "/World/Moon");
            scene.Set("/World/Moon", tokens::kBodyRadius, radius);
            scene.Set("/World/Moon", tokens::kTransform, at);
        } else if (moon == Moon::Sphere) {
            ernie::Sphere::Define(scene, "/World/Moon")
                .SetRadius(radius);
            scene.Set("/World/Moon", tokens::kTransform, at);
        }
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const std::size_t c = kVarSize / 2;
        return std::pair{SpectrumAt(outputs, c, c),
                         SpectrumAt(outputs, c, c + kVarSize / 16)};
    };
    const auto [clearUmbra, clearPenumbra] = view(Moon::None);
    const auto [bodyUmbra, bodyPenumbra] = view(Moon::Body);
    const auto [sphereUmbra, spherePenumbra] = view(Moon::Sphere);
    for (const std::size_t band : {1u, 8u, 14u}) {
        EXPECT_LT(bodyUmbra[band], 0.1 * clearUmbra[band])
            << "band " << band;
        EXPECT_LT(sphereUmbra[band], 0.1 * clearUmbra[band])
            << "band " << band;
        EXPECT_LT(bodyPenumbra[band], 0.95 * clearPenumbra[band])
            << "band " << band;
        EXPECT_NEAR(spherePenumbra[band] / bodyPenumbra[band], 1.0,
                    0.05)
            << "band " << band << ": clear " << clearPenumbra[band]
            << ", analytic " << bodyPenumbra[band] << ", geometric "
            << spherePenumbra[band];
    }
}

// Under a Sphere Moon the sky darkens as under the analytic eclipse's
// CelestialBody: the air the sky's light comes from lies in the
// long-range view's penumbra too, out to where the sun shines on it
// past the Moon. Partly before the sun, by as much; in totality, to
// almost nothing.
TEST_F(RasterDelegateTest,
       ASphereMoonDarkensTheSkyAsTheAnalyticEclipse) {
    constexpr double kElevation = 40.0;
    const double e = kElevation * kPi / 180.0;
    const glm::dvec3 towards(std::cos(e), 0.0, std::sin(e));
    const glm::dvec3 across =
        glm::normalize(glm::cross(towards, {0.0, 1.0, 0.0}));
    const double sunRadius = 0.5 * 0.53 * kPi / 180.0;
    enum class Moon { None, Body, Sphere };
    const auto sky = [&](Moon moon, double radius, double offset) {
        Scene scene =
            SkyScene({0.0, 0.0, 2.0}, {-10.0, 0.0, 3.0}, kElevation);
        ernie::DistantLight(scene, "/World/Sun")
            .SetShadowEnable(true);
        const glm::dmat4 at = glm::translate(
            glm::dmat4(1.0),
            kMoonDistance * (std::cos(offset) * towards +
                             std::sin(offset) * across));
        if (moon == Moon::Body) {
            ernie::CelestialBody::Define(scene, "/World/Moon");
            scene.Set("/World/Moon", tokens::kBodyRadius, radius);
            scene.Set("/World/Moon", tokens::kTransform, at);
        } else if (moon == Moon::Sphere) {
            ernie::Sphere::Define(scene, "/World/Moon")
                .SetRadius(radius);
            scene.Set("/World/Moon", tokens::kTransform, at);
        }
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        return SpectrumAt(outputs, kVarSize / 2, kVarSize / 2);
    };
    const std::vector<double> clear = sky(Moon::None, 0.0, 0.0);
    const double partial = 0.6 * sunRadius;
    const std::vector<double> analytic =
        sky(Moon::Body, kMoonRadius, partial);
    const std::vector<double> geometric =
        sky(Moon::Sphere, kMoonRadius, partial);
    const std::vector<double> total =
        sky(Moon::Sphere, 1.1 * kMoonRadius, 0.0);
    for (const std::size_t band : {1u, 8u, 14u}) {
        EXPECT_LT(analytic[band], 0.8 * clear[band])
            << "band " << band;
        EXPECT_NEAR(geometric[band] / analytic[band], 1.0, 0.05)
            << "band " << band << ": clear " << clear[band]
            << ", analytic " << analytic[band] << ", geometric "
            << geometric[band];
        EXPECT_LT(total[band], 1e-3 * clear[band]) << "band " << band;
    }
}

// A full Moon behind the camera's back to the sun shows its disc at
// Lommel-Seeliger's radiance, its normal albedo times the sun's
// irradiance over pi all across it, sharp against black space.
TEST_F(RasterDelegateTest, AFullMoonShowsItsNormalAlbedo) {
    const glm::dvec3 towards(0.0, 0.0, -1.0);
    Scene scene = FlatScene();
    scene.Remove("/World/Terrain");
    // The sun straight up behind the camera, which looks down at the
    // Moon below, 0.1 degree off the sun's opposition so the camera
    // does not hide it.
    MoonNear(scene, towards, 0.1 * kPi / 180.0);
    scene.Set("/World/Moon", tokens::kBodyAlbedo, glm::dvec3(0.12));
    // 1 mm of sensor at 50 mm: 1.15 degrees across, the half-degree
    // Moon some 14 pixels wide.
    scene.Set(
        "/World/Camera", tokens::kTransform,
        ernie::LookAtTransform({0.0, 0.0, 0.0},
                               scene.WorldTransform("/World/Moon")[3],
                               {0.0, 1.0, 0.0}));
    ernie::Camera(scene, "/World/Camera")
        .SetFocalLength(50.0)
        .SetHorizontalAperture(1.0)
        .SetVerticalAperture(1.0)
        .SetClippingRange(glm::dvec2(1.0, 1e3));
    RasterDelegate delegate(*context_);
    ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
    const ernie::SpectralBands bands;
    const std::vector<float> sun =
        ernie::SunIrradianceOf(scene, "/World/Sun", bands);
    for (const std::size_t row :
         {std::size_t{kVarSize / 2 - 4}, std::size_t{kVarSize / 2}}) {
        const std::vector<double> disc =
            SpectrumAt(outputs, row, kVarSize / 2);
        for (const std::size_t band :
             {std::size_t{2}, std::size_t{12}}) {
            EXPECT_NEAR(
                disc[band] /
                    (0.12 * static_cast<double>(sun[band]) / kPi),
                1.0, 0.02)
                << "row " << row << " band " << band;
        }
    }
    for (const double value : SpectrumAt(outputs, 0, 0)) {
        EXPECT_EQ(value, 0.0);
    }
}

// Seen through the air, the Moon covering the middle of the sun
// leaves the bright ring of an annular eclipse: the irradiance of the
// sun's disc in the image falls to the share of the limb-darkened
// disc the Moon leaves uncovered.
TEST_F(RasterDelegateTest, AnAnnularEclipseLeavesTheSunsRing) {
    constexpr double kElevation = 30.0;
    const double e = kElevation * kPi / 180.0;
    const glm::dvec3 towards(std::cos(e), 0.0, std::sin(e));
    const auto discIrradiance = [&](bool moon) {
        Scene scene =
            SkyScene({0.0, 0.0, 2.0},
                     glm::dvec3(0.0, 0.0, 2.0) + towards, kElevation);
        scene.Set("/World/Air", tokens::kAtmosphereHazeOpticalDepth,
                  0.0);
        ernie::Camera(scene, "/World/Camera")
            .SetFocalLength(50.0)
            .SetHorizontalAperture(1.0)
            .SetVerticalAperture(1.0);
        if (moon) {
            MoonNear(scene, towards, 0.0);
            // At apogee, smaller than the sun.
            scene.Set(
                "/World/Moon", tokens::kTransform,
                glm::translate(glm::dmat4(1.0), 405500e3 * towards));
        }
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        const double pixel = 1.0 / kVarSize / 50.0;
        std::vector<double> irradiance(16, 0.0);
        for (const std::size_t band :
             {std::size_t{2}, std::size_t{12}}) {
            const std::size_t last = kVarSize - 1;
            double background = 0.0;
            for (const std::size_t row : {std::size_t{0}, last}) {
                for (const std::size_t col : {std::size_t{0}, last}) {
                    background +=
                        SpectrumAt(outputs, row, col)[band] / 4.0;
                }
            }
            for (std::uint32_t row = 0; row < kVarSize; ++row) {
                for (std::uint32_t col = 0; col < kVarSize; ++col) {
                    const double x = (col + 0.5) - kVarSize / 2.0;
                    const double y = (row + 0.5) - kVarSize / 2.0;
                    if (x * x + y * y < 10.0 * 10.0) {
                        irradiance[band] +=
                            (SpectrumAt(outputs, row, col)[band] -
                             background) *
                            pixel * pixel;
                    }
                }
            }
        }
        return irradiance;
    };
    const std::vector<double> full = discIrradiance(false);
    const std::vector<double> ring = discIrradiance(true);
    const ernie::SpectralBands bands;
    const double sunRadius = 0.5 * 0.53 * kPi / 180.0;
    const double moonRadius = std::asin(kMoonRadius / 405500e3);
    for (const std::size_t band : {std::size_t{2}, std::size_t{12}}) {
        const double expected =
            EclipseReference(sunRadius, moonRadius, 0.0,
                             bands.Centre(static_cast<int>(band)));
        EXPECT_NEAR(ring[band] / full[band], expected, 0.03)
            << "band " << band;
    }
}

namespace {

/// The planet's shadow by the shadow table's integral, finely, with
/// the CPU's traces: the sun of angular radius `sunRadius` past the
/// atmosphere's disc (`EclipseReference`) and through it, each
/// grazing height's ray bent twice its trace's bending from level at
/// the grazing radius and dimmed by the square of its transmittance
/// from there, its source on the sun's limb-darkened disc.
double PlanetShadowReference(const BandedAtmosphere& air,
                             double distance, double sigma, int band,
                             double sunRadius) {
    const double wavelength = air.bands.Centre(band);
    const double bottom = air.bottomRadius;
    const double top = bottom + air.topAltitude;
    double light = EclipseReference(
        sunRadius, std::asin(top / distance), sigma, wavelength);
    const double alpha0 = std::max(-0.023 + 292.0 / wavelength, 0.0);
    const double halfSun = std::sin(0.5 * sunRadius);
    const double solidAngle = 4.0 * kPi * halfSun * halfSun;
    constexpr int kHeights = 400;
    constexpr int kAzimuths = 200;
    for (int i = 0; i < kHeights; ++i) {
        const double t = (i + 0.5) / kHeights;
        const double b = bottom + (top - bottom) * t * t;
        const double db = (top - bottom) * 2.0 * t / kHeights;
        const double alpha = std::asin(b / distance);
        const double dalpha =
            db / std::sqrt(distance * distance - b * b);
        double grazing = b;
        for (int k = 0; k < 4; ++k) {
            grazing = b / air.RefractiveIndex(grazing - bottom, band);
        }
        if (grazing <= bottom) {
            continue;
        }
        const double bending =
            2.0 * air.Refract(grazing, 0.0, band, 50.0).deviation;
        const double through =
            air.TransmittanceToTop(grazing, 0.0, band);
        const double source = std::abs(alpha - bending);
        const double halfOff = std::sin(0.5 * (source - sigma));
        const double across = std::sin(source) * std::sin(sigma);
        const double room = halfSun * halfSun - halfOff * halfOff;
        if (room <= 0.0) {
            continue;
        }
        const double psiMax =
            across <= room
                ? kPi
                : 2.0 * std::asin(std::sqrt(room / across));
        double radiance = 0.0;
        for (int j = 0; j < kAzimuths; ++j) {
            const double s =
                std::sin(0.5 * (j + 0.5) / kAzimuths * psiMax);
            const double ratio = std::min(
                std::sqrt(halfOff * halfOff + across * s * s) /
                    halfSun,
                1.0);
            radiance +=
                std::pow(std::sqrt(1.0 - ratio * ratio), alpha0);
        }
        radiance *= (alpha0 + 2.0) / (2.0 * solidAngle) * 2.0 *
                    psiMax / kAzimuths;
        light +=
            through * through * radiance * std::sin(alpha) * dalpha;
    }
    return light;
}

}  // namespace

// The planet's shadow table against the reference integral at its
// texels' distances and separations: inside the umbra, where only
// the light refracted through the atmosphere reaches, through the
// penumbra to the full sun.
TEST_F(AtmosphereLutsTest, PlanetShadowMatchesTheReference) {
    ernie::ShaderCompiler compiler;
    AtmosphereLuts luts(*context_, compiler);
    luts.Set(Earth(), {380.0, 780.0, 16}, 0.53f);
    context_->Submit(
        [&](VkCommandBuffer commands) { luts.Record(commands); });
    const std::vector<float> table =
        TableOf(*context_, luts.PlanetShadow());
    const BandedAtmosphere& air = luts.Banded();
    const double top = air.bottomRadius + air.topAltitude;
    const double sunRadius = 0.5 * 0.53 * kPi / 180.0;
    // After the transmittance's four layers, 256 x 64 each.
    for (const int x : {80, 190}) {
        const double distance =
            top * std::exp(UnitOfTexel(x, 256) * std::log(256.0));
        for (const int y : {0, 30, 50, 60}) {
            const double sigma =
                UnitOfTexel(y, 64) *
                (std::asin(top / distance) + sunRadius);
            for (const int band : {2, 13}) {
                const double reference = PlanetShadowReference(
                    air, distance, sigma, band, sunRadius);
                const std::size_t at =
                    (static_cast<std::size_t>(4 + band / 4) * 64 *
                         256 +
                     static_cast<std::size_t>(y) * 256 +
                     static_cast<std::size_t>(x)) *
                        4 +
                    static_cast<std::size_t>(band % 4);
                EXPECT_NEAR(static_cast<double>(table[at]), reference,
                            0.05 * reference + 1e-6)
                    << "texel " << x << ", " << y << " band " << band;
            }
        }
    }
}

// A full Moon at the centre of the Earth's umbra, overhead at
// midnight, glows dim and red: the light the atmosphere refracts into
// the umbra has crossed the long path along the limb twice, where
// Rayleigh scattering took the blue.
TEST_F(RasterDelegateTest, TheEclipsedMoonGlowsRed) {
    const auto moonCentre = [&](bool eclipsed) {
        Scene scene = SkyScene({0.0, 0.0, 2.0}, {0.0, 0.0, 1000.0},
                               eclipsed ? -90.0 : -87.0);
        scene.Set("/World/Air", tokens::kAtmosphereHazeOpticalDepth,
                  0.0);
        scene.Set("/World/Camera", tokens::kTransform,
                  ernie::LookAtTransform({0.0, 0.0, 2.0},
                                         {0.0, 0.0, 1000.0},
                                         {1.0, 0.0, 0.0}));
        ernie::Camera(scene, "/World/Camera")
            .SetFocalLength(50.0)
            .SetHorizontalAperture(1.0)
            .SetVerticalAperture(1.0);
        // Opposite the sun from the Earth's centre, 6371 km below.
        ernie::CelestialBody::Define(scene, "/World/Moon");
        scene.Set("/World/Moon", tokens::kTransform,
                  glm::translate(
                      glm::dmat4(1.0),
                      glm::dvec3(0.0, 0.0, kMoonDistance - 6371e3)));
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        return SpectrumAt(outputs, kVarSize / 2, kVarSize / 2);
    };
    // Uneclipsed, the sun 3 degrees aside, for the full Moon's
    // brightness through the same air.
    const std::vector<double> full = moonCentre(false);
    const std::vector<double> eclipsed = moonCentre(true);
    const double blue = eclipsed[2] / full[2];
    const double red = eclipsed[13] / full[13];
    std::printf("umbra: blue %.3g, red %.3g of the full Moon\n", blue,
                red);
    EXPECT_GT(red, 1e-5);
    EXPECT_LT(red, 1e-2);
    EXPECT_GT(red, 10.0 * blue);
}

TEST_F(RasterDelegateTest, EveryDistantLightLightsTheAir) {
    // The sun up, or set with a second distant light of its own
    // brightness where it was: the air, the sky and the ground see
    // the same light either way.
    const double e = 30.0 * kPi / 180.0;
    const double set = -40.0 * kPi / 180.0;
    const auto render = [&](bool second, bool sky) {
        Scene scene = FlatScene();
        ernie::Atmosphere::Define(scene, "/World/Air");
        const glm::dvec3 up(std::cos(e), 0.0, std::sin(e));
        scene.Set(
            "/World/Sun", tokens::kTransform,
            ernie::DistantLightTransform(
                second ? glm::dvec3(std::cos(set), 0.0, std::sin(set))
                       : up));
        if (second) {
            ernie::DistantLight::Define(scene, "/World/Twin")
                .SetShadowEnable(false)
                .SetTransform(ernie::DistantLightTransform(up));
        }
        // Down at the ground from 2 km, or up at the sky 20 degrees
        // above the horizon, towards the light.
        const double a = 20.0 * kPi / 180.0;
        scene.Set("/World/Camera", tokens::kTransform,
                  sky ? ernie::LookAtTransform(
                            {0.0, 0.0, 2.0},
                            {std::cos(a), 0.0, 2.0 + std::sin(a)},
                            {0.0, 0.0, 1.0})
                      : ernie::LookAtTransform({0.0, 0.0, 2000.0},
                                               {0.0, 0.0, 0.0},
                                               {0.0, 1.0, 0.0}));
        RasterDelegate delegate(*context_);
        ernie::RenderProductSetOutputs outputs = delegate.Step(scene);
        return SpectrumAt(outputs, kVarSize / 2, kVarSize / 2);
    };
    for (const bool sky : {false, true}) {
        const std::vector<double> sun = render(false, sky);
        const std::vector<double> twin = render(true, sky);
        ASSERT_EQ(sun.size(), twin.size());
        for (std::size_t b = 0; b < sun.size(); ++b) {
            EXPECT_GT(sun[b], 0.0);
            EXPECT_NEAR(twin[b] / sun[b], 1.0, 0.03)
                << (sky ? "sky" : "ground") << ", band " << b;
        }
    }
}

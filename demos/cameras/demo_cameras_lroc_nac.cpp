// SPDX-License-Identifier: BSD-3-Clause
// LROC's Narrow Angle Camera line-scanning the Moon's south pole at a
// low Sun, posed from a real ISD on LOLA's terrain (fetched on the
// first run, 12 MB).
//
// References:
// - M. S. Robinson et al., "Lunar Reconnaissance Orbiter Camera
//   (LROC) Instrument Overview", Space Science Reviews 150:81-124,
//   2010: Table 2 (the NACs) and Appendix A (their companding).
// - USGS Astrogeology, usgscsm's test data (toughLroNacLineScan.json,
//   an ISD ALE wrote for a NAC image) and ALE,
//   https://github.com/DOI-USGS/usgscsm and
//   https://github.com/DOI-USGS/ale
// - D. E. Smith et al., "The Lunar Orbiter Laser Altimeter
//   Investigation on the Lunar Reconnaissance Orbiter Mission", Space
//   Science Reviews 150:209-241, 2010 (LOLA, whose polar LDEM is the
//   terrain).
// - G. Kopp and J. L. Lean, "A new, lower value of total solar
//   irradiance", Geophysical Research Letters 38:L01706, 2011 (1361
//   W/m^2 at 1 au).

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/dem.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/isd.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "util.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kPi = std::numbers::pi;
constexpr double kDegree = kPi / 180.0;

/// The image lines rendered: 8 strips of 64 about the image's middle,
/// where it passes the pole.
constexpr std::uint32_t kFirstLine = 3840;
constexpr std::uint32_t kStripLines = 64;
constexpr std::uint32_t kStrips = 8;

/// The terrain about the middle line's ground: cells of 5 m, the
/// DEM's own, 12 km across.
constexpr std::uint32_t kCells = 2401;
constexpr double kCellSize = 5.0;

/// The Sun's radius and the astronomical unit, metres; the total
/// solar irradiance at 1 au, W/m^2 (Kopp and Lean 2011).
constexpr double kSunRadius = 6.957e8;
constexpr double kAu = 1.495978707e11;
constexpr double kSolarConstant = 1361.0;

/// NAC-L's published values (Robinson et al. 2010, Table 2; section
/// 3.1.2 for the KLI-5001G's QE): focal ratio, gain, detector noise
/// and full well. The ISD sets the geometry and line time.
constexpr double kNacFNumber = 3.577;
constexpr double kNacElectronsPerDn = 90.5;
constexpr double kNacReadNoise = 101.0;
constexpr double kNacFullWell = 334000.0;

/// The NAC's nominal 12- to 8-bit table (Appendix A, code 0): five
/// linear segments.
double NacCompand(double x) {
    return x < 32     ? std::floor(x / 2)
           : x < 136  ? std::floor(x / 4 + 8)
           : x < 543  ? std::floor(x / 8 + 25)
           : x < 2207 ? std::floor(x / 16 + 59)
                      : std::floor(x / 32 + 128);
}

// ---------------------------------------------------------------
// LOLA

/// LOLA's 5 m polar LDEM south of 87.5 S at the PDS Geosciences Node:
/// 30336 x 30336 16-bit samples, the pole at line and sample 15167.5.
constexpr const char* kLdemUrl =
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/lola_gdr/polar/img/ldem_875s_5m.img";
constexpr std::uint64_t kLdemSize = 30336;
constexpr double kLdemOffset = 15167.5;
constexpr double kLdemScale = 5.0;
constexpr double kMoonRadius = 1737400.0;

/// The cached window's label.
std::filesystem::path LdemLabel() {
    return demos::CacheDir() / "lola" / "ldem_875s_5m_lroc_nac.lbl";
}

/// The polar grid's fractional (line, sample) of a latitude and east
/// longitude (`PolarStereographicDem::ToGrid`).
glm::dvec2 LdemGrid(double latitudeDeg, double longitudeDeg) {
    PolarStereographicDem grid;
    grid.mapScale = kLdemScale;
    grid.lineProjectionOffset = kLdemOffset;
    grid.sampleProjectionOffset = kLdemOffset;
    grid.referenceRadius = kMoonRadius;
    return grid.ToGrid(latitudeDeg, longitudeDeg);
}

/// Fetches the window of the polar LDEM `half` pixels about latitude
/// `latitudeDeg` and longitude `longitudeDeg`, 256 lines a request,
/// each line's samples in the window alone (HTTP range requests; the
/// product is 1.8 GB), and writes it to the cache as an image and a
/// PDS3 label of its own, which `ReadPolarStereographicDem` reads as
/// it would the whole product. As demo_cameras_lroc_nac.py's
/// `fetch_ldem_window`.
void FetchLdemWindow(double latitudeDeg, double longitudeDeg,
                     std::uint64_t half) {
    const glm::dvec2 centre = LdemGrid(latitudeDeg, longitudeDeg);
    const auto firstLine =
        static_cast<std::uint64_t>(std::lround(centre.x)) - half;
    const auto firstSample =
        static_cast<std::uint64_t>(std::lround(centre.y)) - half;
    const std::uint64_t size = 2 * half + 1;
    std::printf("Fetching %llu lines of %llu samples from %s\n",
                static_cast<unsigned long long>(size),
                static_cast<unsigned long long>(size), kLdemUrl);
    std::string rows;
    const std::uint64_t rowBytes = 2 * kLdemSize;
    for (std::uint64_t start = 0; start < size; start += 256) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        for (std::uint64_t line = firstLine + start;
             line < firstLine + std::min(start + 256, size); ++line) {
            ranges.emplace_back(line * rowBytes + 2 * firstSample,
                                line * rowBytes +
                                    2 * (firstSample + size) - 1);
        }
        for (const std::string& part :
             demos::FetchRanges(kLdemUrl, ranges)) {
            rows += part;
        }
    }
    const std::filesystem::path label = LdemLabel();
    std::filesystem::create_directories(label.parent_path());
    {
        std::ofstream image(label.parent_path() /
                                "ldem_875s_5m_lroc_nac.img",
                            std::ios::binary);
        image.write(rows.data(),
                    static_cast<std::streamsize>(rows.size()));
    }
    std::filesystem::path part = label;
    part += ".part";
    {
        std::ofstream text(part);
        text << "PDS_VERSION_ID = PDS3\n"
                "/* A window of LOLA's LDEM_875S_5M "
                "(LRO-L-LOLA-4-GDR-V1.0), fetched\n   by "
                "demos/cameras/demo_cameras_lroc_nac from the PDS "
                "Geosciences Node. */\n"
                "RECORD_TYPE = FIXED_LENGTH\n"
             << "RECORD_BYTES = " << 2 * size << "\n"
             << "FILE_RECORDS = " << size << "\n"
             << "^IMAGE = \"LDEM_875S_5M_LROC_NAC.IMG\"\n"
                "OBJECT = IMAGE\n"
             << "  LINES = " << size << "\n"
             << "  LINE_SAMPLES = " << size << "\n"
             << "  SAMPLE_TYPE = LSB_INTEGER\n"
                "  SAMPLE_BITS = 16\n"
                "  UNIT = METER\n"
                "  SCALING_FACTOR = 0.5\n"
                "  OFFSET = 1737400.\n"
                "END_OBJECT = IMAGE\n"
                "OBJECT = IMAGE_MAP_PROJECTION\n"
                "  MAP_PROJECTION_TYPE = \"POLAR STEREOGRAPHIC\"\n"
                "  A_AXIS_RADIUS = 1737.4 <km>\n"
                "  CENTER_LATITUDE = -90 <deg>\n"
                "  CENTER_LONGITUDE = 0 <deg>\n"
                "  MAP_SCALE = 5 <m/pix>\n"
             << "  LINE_PROJECTION_OFFSET = "
             << kLdemOffset - static_cast<double>(firstLine)
             << " <pix>\n"
             << "  SAMPLE_PROJECTION_OFFSET = "
             << kLdemOffset - static_cast<double>(firstSample)
             << " <pix>\n"
                "END_OBJECT = IMAGE_MAP_PROJECTION\n"
                "END\n";
    }
    std::filesystem::rename(part, label);
    std::printf("Wrote %s (%.1f MB)\n", label.string().c_str(),
                double(rows.size()) / 1e6);
}

// ------------------------------------------------------------ the
// demo

/// Product `product`'s var `var` on the host.
Tensor Var(RenderProductSetOutputs& outputs, const char* product,
           const char* var) {
    return outputs.at(product)
        .vars.at(var)
        .Map(MapDevice::Cpu)
        .tensor;
}

/// Latitude and east longitude (degrees) of body-fixed `point`.
glm::dvec2 LatitudeLongitude(const glm::dvec3& point) {
    const glm::dvec3 d = glm::normalize(point);
    return {std::asin(std::clamp(d.z, -1.0, 1.0)) / kDegree,
            std::atan2(d.y, d.x) / kDegree};
}

class LrocNacDemo : public app::Demo {
  public:
    LrocNacDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options),
          nac_(ReadIsdFile(
              demos::DataFile("toughLroNacLineScan.json").string())) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // The middle line's ground, on the reference sphere.
        const double middle =
            kFirstLine + 0.5 * kStripLines * kStrips;
        const double samples = nac_.Isd().samples;
        const glm::dvec3 ground =
            nac_.ImageToGround(middle, 0.5 * samples, 0.0);
        const glm::dvec2 site = LatitudeLongitude(ground);
        // LOLA's terrain there, else the bare sphere, in a frame
        // there: x east, y north, z up. Fetched once, but not for the
        // checks, which run offline.
        const std::filesystem::path label = LdemLabel();
        if (!std::filesystem::exists(label) && !options.test &&
            demos::CanDownload()) {
            try {
                FetchLdemWindow(
                    site.x, site.y,
                    static_cast<std::uint64_t>(std::lround(
                        (kCells - 1) / 2 * kCellSize / kLdemScale)) +
                        40);
            } catch (const std::exception& error) {
                std::printf("%s\n", error.what());
            }
        }
        dem_.referenceRadius = kMoonRadius;
        if (std::filesystem::exists(label)) {
            dem_ = ReadPolarStereographicDem(label.string());
        } else {
            std::printf("No LOLA terrain in %s; the bare sphere "
                        "stands in.\n",
                        label.parent_path().string().c_str());
        }
        const LocalTerrain terrain =
            LocalTerrainOf(dem_, site.x, site.y, kCells, kCellSize);
        bodyToLocal_ = terrain.bodyToLocal;
        // The regolith's normal albedo, about 0.1, flat (its
        // reddening left out).
        Material::Define(scene, "/World/Looks/Regolith")
            .SetReflectance(
                std::vector<float>{250.0f, 0.12f, 800.0f, 0.12f});
        Terrain::Define(scene, "/World/Terrain")
            .SetHeights(terrain.heights)
            .SetOrigin(terrain.origin)
            .SetCellSize(terrain.cellSize)
            .SetMaterialBinding(Path("/World/Looks/Regolith"));

        // The Sun when the middle line was taken.
        const double taken = nac_.ImageTime(middle);
        const glm::dvec3 toSun = nac_.SunPosition(taken) - ground;
        const double distance = glm::length(toSun);
        const glm::dvec3 local =
            glm::dmat3(bodyToLocal_) * (toSun / distance);
        sunElevation_ = std::asin(local.z) / kDegree;
        std::printf("%.3f S, %.2f E; the Sun %.4f au away, %.2f "
                    "degrees up\n",
                    -site.x, site.y, distance / kAu, sunElevation_);
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(kSolarConstant * (kAu / distance) *
                           (kAu / distance))
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(2.0 * std::asin(kSunRadius / distance) /
                      kDegree);
        scene.Set("/World/Sun", tokens::kTransform,
                  DistantLightTransform(local));

        DefineNac();
        // The viewer's own view: the site from 3 km up to the
        // south-west of it, the Sun low behind.
        Camera::Define(scene, "/World/View")
            .SetFocalLength(24.0)
            .SetClippingRange(glm::dvec2(10.0, 1e5))
            // Sunlit regolith's exposure: the Looney 11 rule, f/11 at
            // 1/ISO seconds.
            .SetExposureFStop(11.0)
            .SetExposureIso(100.0)
            .SetExposureTime(1.0 / 100.0);
        scene.Set("/World/View", tokens::kTransform,
                  LookAtTransform({-4e3, -4e3, 3e3}, {0.0, 0.0, 0.0},
                                  {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/View"))
            .SetResolution(glm::dvec2(480, 320))
            .SetAmbient(0.0);
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/View");
        }
    }

    /// The NAC's 8-bit image of the lines, as it sends it.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, Image());
    }

    void TestFinal() override {
        Camera(GetScene(), "/World/Nac").SetSensorNoise(false);
        CheckStripsSeeTheGroundWhereTheIsdPutsIt();
        CheckCompanding();
        Camera(GetScene(), "/World/Nac").SetSensorNoise(true);
    }

  private:
    /// The NAC as published, its geometry the ISD's: the KLI-5001G's
    /// QE (> 60% over 400 to 760 nm, section 3.1.2), gain, noise and
    /// full well, a 12-bit ADC, the even and odd channels, the
    /// 12- to 8-bit table and photographic outputs to show.
    void DefineNac() {
        Scene& scene = GetScene();
        Camera::Define(scene, "/World/Nac")
            .SetSensorType("mono")
            .SetSensorQeMono(std::vector<float>{390.0f, 0.0f, 400.0f,
                                                0.6f, 760.0f, 0.6f,
                                                770.0f, 0.0f})
            .SetExposureFStop(kNacFNumber)
            .SetExposureIso(100.0)  // the gain is the ADC's
            .SetSensorElectronsPerDn(kNacElectronsPerDn)
            .SetSensorReadNoise(kNacReadNoise)
            .SetSensorFullWell(kNacFullWell)
            .SetSensorAdcBits(12)
            .SetSensorBlackLevel(0.0)
            .SetSensorNoise(true);
        UseStrip(0, false);
        std::vector<Path> vars;
        for (const char* var :
             {"RawDN", "Electrons", "PrimId", "LdrColor"}) {
            const Path path = Path("/Render/Nac").Child(var);
            RenderVar::Define(scene, path).SetSourceName(var);
            vars.push_back(path);
        }
        std::vector<Path> filters;
        const char* types[] = {"qe",          "noise",        "adc",
                               "readoutTaps", "compandTable", "cie",
                               "exposure"};
        for (std::size_t i = 0; i < std::size(types); ++i) {
            const Path path = Path("/Render/Nac/Chain")
                                  .Child("Step" + std::to_string(i));
            SensorFilterPrim::Define(scene, path)
                .SetFilterType(types[i]);
            filters.push_back(path);
        }
        scene.Set(filters[3], "inputs:taps", 2.0);
        RenderProduct::Define(scene, "/Render/Nac")
            .SetCamera(Path("/World/Nac"))
            .SetResolution(
                glm::dvec2(nac_.Isd().samples, kStripLines))
            .SetOrderedVars(vars)
            .SetFilters(filters);
    }

    /// Camera /World/Nac as strip `strip` of the lines, posed at its
    /// end (or its start) in the local frame.
    void UseStrip(std::uint32_t strip, bool atEnd) {
        const IsdLineScanner::Strip s = nac_.StripOf(
            kFirstLine + strip * kStripLines, kStripLines);
        CameraParams camera = s.camera;
        camera.transform = bodyToLocal_ * (atEnd ? s.to : s.from);
        AuthorIsdCamera(GetScene(), "/World/Nac", camera);
    }

    /// Strip `strip` rendered: the camera stepped at its start, then
    /// at its end, the rows its lines.
    RenderProductSetOutputs RenderStrip(std::uint32_t strip) {
        const Path products[] = {"/Render/Nac"};
        UseStrip(strip, false);
        GetRenderer().Step(GetScene(), products);
        UseStrip(strip, true);
        return GetRenderer().Step(GetScene(), products);
    }

    /// The 8-bit image of the lines, uint8 (lines, samples, 4) grey.
    Tensor Image() {
        const std::uint32_t samples = nac_.Isd().samples;
        const std::uint32_t lines = kStripLines * kStrips;
        std::vector<std::uint8_t> pixels(
            std::size_t(lines) * samples * 4, 255);
        for (std::uint32_t k = 0; k < kStrips; ++k) {
            RenderProductSetOutputs outputs = RenderStrip(k);
            const Tensor raw = Var(outputs, "/Render/Nac", "RawDN");
            // A mono sensor's numbers are each pixel's first channel.
            const std::int64_t channels =
                raw.Elements() /
                (std::int64_t(kStripLines) * samples);
            for (std::uint32_t r = 0; r < kStripLines; ++r) {
                for (std::uint32_t s = 0; s < samples; ++s) {
                    const auto dn = static_cast<std::uint8_t>(
                        raw.As<std::uint16_t>()[(r * samples + s) *
                                                channels]);
                    std::uint8_t* texel =
                        &pixels[((std::size_t(k) * kStripLines + r) *
                                     samples +
                                 s) *
                                4];
                    texel[0] = texel[1] = texel[2] = dn;
                }
            }
        }
        return Tensor::Copy<std::uint8_t>(
            pixels, {std::int64_t(lines), std::int64_t(samples), 4});
    }

    /// Balls on the ground (lifted 20 m up the pixels' rays, clear of
    /// the terrain's cells) where the ISD's model puts three pixels'
    /// rays on LOLA's terrain show on those pixels of their strips:
    /// the ISD's geometry, as Ernie draws it in the local frame.
    void CheckStripsSeeTheGroundWhereTheIsdPutsIt() {
        Scene& scene = GetScene();
        const std::uint32_t samples = nac_.Isd().samples;
        const std::array<std::array<std::uint32_t, 2>, 3> pixels = {
            {{kFirstLine + 100, 300},
             {kFirstLine + 256, 1266},
             {kFirstLine + 400, 2200}}};
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            const auto [line, sample] = pixels[i];
            const double l = line + 0.5;
            const double s = sample + 0.5;
            double h = 0.0;
            glm::dvec3 ground(0.0);
            for (int k = 0; k < 4; ++k) {
                ground = nac_.ImageToGround(l, s, h);
                const glm::dvec2 at = LatitudeLongitude(ground);
                h = dem_.HeightAt(at.x, at.y);
            }
            const glm::dvec3 eye = nac_.Position(nac_.ImageTime(l));
            const glm::dvec3 lifted =
                ground + 20.0 * glm::normalize(eye - ground);
            const Path ball("/World/Marker" + std::to_string(i));
            // About two lines and 40 samples across.
            Sphere::Define(scene, ball).SetRadius(15.0);
            scene.Set(
                ball, tokens::kTransform,
                glm::translate(glm::dmat4(1.0),
                               glm::dvec3(bodyToLocal_ *
                                          glm::dvec4(lifted, 1.0))));
            const std::uint32_t strip =
                (line - kFirstLine) / kStripLines;
            RenderProductSetOutputs outputs = RenderStrip(strip);
            const Tensor ids = Var(outputs, "/Render/Nac", "PrimId");
            glm::dvec2 sum(0.0);
            int hits = 0;
            for (std::uint32_t r = 0; r < kStripLines; ++r) {
                for (std::uint32_t c = 0; c < samples; ++c) {
                    if (GetRenderer().PrimPath(
                            ids.As<std::uint32_t>()[r * samples +
                                                    c]) == ball) {
                        sum += glm::dvec2(r + 0.5, c + 0.5);
                        ++hits;
                    }
                }
            }
            scene.Remove(ball);
            Check(hits > 0, "each marker should show in its strip");
            const glm::dvec2 centre = sum / double(hits);
            const double row =
                l - kFirstLine - double(strip * kStripLines);
            std::printf("marker %zu: (%.2f, %.2f) for (%.2f, %.2f)\n",
                        i, centre.x, centre.y, row, s);
            Check(std::abs(centre.x - row) < 1.0 &&
                      std::abs(centre.y - s) < 1.5,
                  "a marker should land on its pixel");
        }
    }

    /// Every pixel's 8-bit number is the table's of its 12-bit one,
    /// its electrons over the gain (either side of a half DN, which
    /// the GPU's floats may round either way); some of the ground is
    /// lit.
    void CheckCompanding() {
        RenderProductSetOutputs outputs = RenderStrip(kStrips / 2);
        const Tensor electrons =
            Var(outputs, "/Render/Nac", "Electrons");
        const Tensor raw = Var(outputs, "/Render/Nac", "RawDN");
        std::int64_t lit = 0;
        for (std::int64_t i = 0; i < raw.Elements(); ++i) {
            const double e = std::min<double>(
                electrons.As<float>()[i], kNacFullWell);
            const auto code = [&](double scale) {
                return NacCompand(std::clamp(
                    std::floor(e * scale / kNacElectronsPerDn + 0.5),
                    0.0, 4095.0));
            };
            const std::uint16_t dn = raw.As<std::uint16_t>()[i];
            Check(dn == code(1.0 - 1e-6) || dn == code(1.0 + 1e-6),
                  "the NAC should send its table's 8-bit numbers");
            lit += e > 1000.0 ? 1 : 0;
        }
        std::printf("NAC: %lld lit pixels of %lld\n",
                    static_cast<long long>(lit),
                    static_cast<long long>(raw.Elements()));
        Check(lit > 0, "the low Sun should light some of the ground");
    }

    IsdLineScanner nac_;
    PolarStereographicDem dem_;
    glm::dmat4 bodyToLocal_{1.0};
    double sunElevation_ = 0.0;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    LrocNacDemo, "cameras", "lroc_nac",
    "LROC's Narrow Angle Camera over the Moon's south "
    "pole as an ISD has it, on LOLA's polar terrain "
    "under a Sun 1 degree up.");

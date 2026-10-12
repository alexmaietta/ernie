// SPDX-License-Identifier: BSD-3-Clause
// LROC's Narrow Angle Camera scanning Tycho's central peak as image
// M1282557272RE's ISD has it, on LROC's 2 m stereo terrain (fetched
// on the first run).
//
// References:
// - M. S. Robinson et al., "Lunar Reconnaissance Orbiter Camera
//   (LROC) Instrument Overview", Space Science Reviews 150:81-124,
//   2010: Table 2 (the NACs) and Appendix A (their companding).
// - USGS Astrogeology, ISIS and ALE (the ISD, from the image's EDR
//   with SPICE from the USGS web service),
//   https://github.com/DOI-USGS/ISIS3 and
//   https://github.com/DOI-USGS/ale
// - M. R. Henriksen et al., "Extracting accurate and precise
//   topography from LROC narrow angle camera stereo observations",
//   Icarus 283:122-137, 2017 (the NAC DTMs; NAC_DTM_TYCHOPK).
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
#include <cstring>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kPi = std::numbers::pi;
constexpr double kDegree = kPi / 180.0;

/// The image lines rendered: 8 strips of 256 near the image's
/// southern end, beside the peak.
constexpr std::uint32_t kFirstLine = 22528;
constexpr std::uint32_t kStripLines = 256;
constexpr std::uint32_t kStrips = 8;

/// The terrain about the middle line's ground: cells of 2 m, the
/// DTM's own, 3.2 km across.
constexpr std::uint32_t kCells = 1601;
constexpr double kCellSize = 2.0;
constexpr double kMoonRadius = 1737400.0;

/// The Sun's radius and the astronomical unit, metres; the total
/// solar irradiance at 1 au, W/m^2 (Kopp and Lean 2011).
constexpr double kSunRadius = 6.957e8;
constexpr double kAu = 1.495978707e11;
constexpr double kSolarConstant = 1361.0;

/// The NACs' focal ratio, gain, noise and full well (Robinson et al.
/// 2010, Table 2).
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

// ----------------------------------------------------------------
// DTM

/// LROC's NAC DTM mosaic of the peak: a GeoTIFF of 32-bit reals, one
/// row a strip, as its PDS3 label gives it.
constexpr const char* kDtmUrl =
    "https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_"
    "orbiter/pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/DATA/SDP/NAC_DTM/"
    "TYCHOPK/NAC_DTM_TYCHOPK.TIF";
constexpr std::uint64_t kDtmLines = 30892;
constexpr std::uint64_t kDtmSamples = 12240;
constexpr double kDtmPixelsPerDegree = 15161.675212075;
constexpr double kDtmParallel = -43.3;
constexpr double kDtmNorth = -42.3668223;
constexpr double kDtmWest = 348.0633983;

/// The cached window's label.
std::filesystem::path DtmLabel() {
    return demos::CacheDir() / "lroc" / "nac_dtm_tychopk_nac.lbl";
}

/// The bytes of `url` from `first` for `count`, little-endian.
std::string Bytes(const std::string& url, std::uint64_t first,
                  std::uint64_t count) {
    const std::pair<std::uint64_t, std::uint64_t> range[] = {
        {first, first + count - 1}};
    return demos::FetchRanges(url, range).front();
}

template <typename T>
T Read(const std::string& bytes, std::size_t at) {
    T value{};
    std::memcpy(&value, bytes.data() + at, sizeof(T));
    return value;
}

/// The byte offset of every row of a little-endian TIFF of one row a
/// strip (its StripOffsets, tag 273).
std::vector<std::uint64_t> RowOffsets(const std::string& url,
                                      std::uint64_t firstRow,
                                      std::uint64_t rows) {
    const std::string header = Bytes(url, 0, 8);
    if (header.substr(0, 2) != "II") {
        throw std::runtime_error(url +
                                 " is not a little-endian TIFF");
    }
    const auto ifd = Read<std::uint32_t>(header, 4);
    const auto entries = Read<std::uint16_t>(Bytes(url, ifd, 2), 0);
    const std::string table = Bytes(url, ifd + 2, 12u * entries);
    for (std::uint16_t e = 0; e < entries; ++e) {
        if (Read<std::uint16_t>(table, 12u * e) != 273) {
            continue;
        }
        const auto type = Read<std::uint16_t>(table, 12u * e + 2);
        const auto at = Read<std::uint32_t>(table, 12u * e + 8);
        const std::uint64_t size = type == 16 ? 8 : 4;
        const std::string offsets =
            Bytes(url, at + size * firstRow, size * rows);
        std::vector<std::uint64_t> result;
        for (std::uint64_t r = 0; r < rows; ++r) {
            result.push_back(
                size == 8 ? Read<std::uint64_t>(offsets, 8 * r)
                          : Read<std::uint32_t>(offsets, 4 * r));
        }
        return result;
    }
    throw std::runtime_error(url + " has no strip offsets");
}

/// Fetches the DTM's window `half` metres about a point into the
/// cache, as an image and a PDS3 label `ReadCylindricalDem` reads.
void FetchDtmWindow(double latitudeDeg, double longitudeDeg,
                    double half) {
    const double along = kDtmPixelsPerDegree;
    const double across = along * std::cos(kDtmParallel * kDegree);
    const double degrees = half / kMoonRadius / kDegree;
    const auto firstRow = static_cast<std::uint64_t>(
        std::floor((kDtmNorth - latitudeDeg - degrees) * along));
    const auto rows =
        static_cast<std::uint64_t>(std::ceil(2.0 * degrees * along));
    const double spread = degrees / std::cos(latitudeDeg * kDegree);
    const auto firstColumn = static_cast<std::uint64_t>(std::floor(
        (std::fmod(longitudeDeg - kDtmWest + 720.0, 360.0) - spread) *
        across));
    const auto columns =
        static_cast<std::uint64_t>(std::ceil(2.0 * spread * across));
    if (firstRow + rows > kDtmLines ||
        firstColumn + columns > kDtmSamples) {
        throw std::runtime_error("The window leaves the DTM");
    }
    std::printf("Fetching %llu rows of %llu samples from %s\n",
                static_cast<unsigned long long>(rows),
                static_cast<unsigned long long>(columns), kDtmUrl);
    const std::vector<std::uint64_t> offsets =
        RowOffsets(kDtmUrl, firstRow, rows);
    // Each run of contiguous rows in one request (the server serves
    // one range a request).
    const std::uint64_t rowBytes = 4 * kDtmSamples;
    std::string data;
    for (std::uint64_t start = 0; start < rows;) {
        std::uint64_t end = start + 1;
        while (end < rows &&
               offsets[end] == offsets[end - 1] + rowBytes) {
            ++end;
        }
        const std::string run =
            Bytes(kDtmUrl, offsets[start], (end - start) * rowBytes);
        for (std::uint64_t r = 0; r < end - start; ++r) {
            data.append(run, r * rowBytes + 4 * firstColumn,
                        4 * columns);
        }
        start = end;
    }
    const std::filesystem::path label = DtmLabel();
    std::filesystem::create_directories(label.parent_path());
    {
        std::ofstream image(label.parent_path() /
                                "nac_dtm_tychopk_nac.img",
                            std::ios::binary);
        image.write(data.data(),
                    static_cast<std::streamsize>(data.size()));
    }
    std::filesystem::path part = label;
    part += ".part";
    {
        std::ofstream text(part);
        text.precision(12);
        text << "PDS_VERSION_ID = PDS3\n"
                "/* A window of LROC's NAC_DTM_TYCHOPK "
                "(LRO-L-LROC-5-RDR-V1.0), fetched\n   by "
                "demos/cameras/demo_cameras_lroc_nac from the PDS. "
                "*/\n"
                "RECORD_TYPE = FIXED_LENGTH\n"
             << "RECORD_BYTES = " << 4 * columns << "\n"
             << "FILE_RECORDS = " << rows << "\n"
             << "^IMAGE = \"NAC_DTM_TYCHOPK_NAC.IMG\"\n"
                "OBJECT = IMAGE\n"
             << "  LINES = " << rows << "\n"
             << "  LINE_SAMPLES = " << columns << "\n"
             << "  SAMPLE_TYPE = PC_REAL\n"
                "  SAMPLE_BITS = 32\n"
                "  UNIT = METER\n"
                "END_OBJECT = IMAGE\n"
                "OBJECT = IMAGE_MAP_PROJECTION\n"
                "  MAP_PROJECTION_TYPE = \"EQUIRECTANGULAR\"\n"
                "  A_AXIS_RADIUS = 1737.4 <KM>\n"
             << "  CENTER_LATITUDE = " << kDtmParallel << " <DEG>\n"
             << "  MAP_RESOLUTION = " << along << " <PIX/DEG>\n"
             << "  MAXIMUM_LATITUDE = "
             << kDtmNorth - double(firstRow) / along << " <DEG>\n"
             << "  WESTERNMOST_LONGITUDE = "
             << kDtmWest + double(firstColumn) / across << " <DEG>\n"
             << "END_OBJECT = IMAGE_MAP_PROJECTION\n"
                "END\n";
    }
    std::filesystem::rename(part, label);
    std::printf("Wrote %s (%.1f MB)\n", label.string().c_str(),
                double(data.size()) / 1e6);
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
              demos::DataFile("lroNacM1282557272RE.json").string())) {
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
        // The NAC DTM there inside LOLA's terrain (the lroc demo's
        // cached window), else the bare sphere, in a frame there: x
        // east, y north, z up. Fetched once, but not for the checks,
        // which run offline.
        const double half = 0.5 * (kCells - 1) * kCellSize + 200.0;
        const std::filesystem::path label = DtmLabel();
        if (!std::filesystem::exists(label) && !options.test &&
            demos::CanDownload()) {
            try {
                FetchDtmWindow(site.x, site.y, half);
            } catch (const std::exception& error) {
                std::printf("%s\n", error.what());
            }
        }
        dtm_.referenceRadius = kMoonRadius;
        lola_.referenceRadius = kMoonRadius;
        if (std::filesystem::exists(label)) {
            dtm_ = ReadCylindricalDem(label.string());
        } else {
            std::printf("No NAC DTM in %s; LOLA's terrain or the "
                        "bare sphere stands in.\n",
                        label.parent_path().string().c_str());
        }
        const std::filesystem::path lola =
            demos::CacheDir() / "lola" / "ldem_512_lroc.lbl";
        if (std::filesystem::exists(lola)) {
            lola_ = ReadCylindricalDem(lola.string());
        }
        const LocalTerrain terrain = LocalTerrainOf(
            dtm_, lola_, site.x, site.y, kCells, kCellSize);
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
        std::printf("%.3f S, %.2f E; the Sun %.4f au away, %.1f "
                    "degrees up\n",
                    -site.x, std::fmod(site.y + 360.0, 360.0),
                    distance / kAu, std::asin(local.z) / kDegree);
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(kSolarConstant * (kAu / distance) *
                           (kAu / distance))
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(2.0 * std::asin(kSunRadius / distance) /
                      kDegree);
        scene.Set("/World/Sun", tokens::kTransform,
                  DistantLightTransform(local));

        DefineNac();
        // The viewer's own view: the flank from 1.5 km up to its
        // south-west.
        Camera::Define(scene, "/World/View")
            .SetFocalLength(24.0)
            .SetClippingRange(glm::dvec2(10.0, 1e5))
            // Sunlit regolith's exposure: the Looney 11 rule, f/11 at
            // 1/ISO seconds.
            .SetExposureFStop(11.0)
            .SetExposureIso(100.0)
            .SetExposureTime(1.0 / 100.0);
        scene.Set("/World/View", tokens::kTransform,
                  LookAtTransform({-1.8e3, -1.8e3, 1.5e3},
                                  {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}));
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
        CheckReadout();
        Camera(GetScene(), "/World/Nac").SetSensorNoise(true);
    }

  private:
    /// The NAC's sensor and readout as published; the ISD sets its
    /// geometry and line time.
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

    /// The terrain's height (m) at body-fixed `point`'s latitude and
    /// longitude: the DTM's, else LOLA's.
    double HeightAt(const glm::dvec3& point) const {
        const glm::dvec2 at = LatitudeLongitude(point);
        const double h = dtm_.HeightAt(at.x, at.y);
        return std::isnan(h) ? lola_.HeightAt(at.x, at.y) : h;
    }

    /// Balls where the ISD puts three pixels' rays on the terrain
    /// show on those pixels.
    void CheckStripsSeeTheGroundWhereTheIsdPutsIt() {
        Scene& scene = GetScene();
        const std::uint32_t samples = nac_.Isd().samples;
        const std::array<std::array<std::uint32_t, 2>, 3> pixels = {
            {{kFirstLine + 100, 600},
             {kFirstLine + 1100, 2532},
             {kFirstLine + 1900, 4400}}};
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            const auto [line, sample] = pixels[i];
            const double l = line + 0.5;
            const double s = sample + 0.5;
            double h = 0.0;
            glm::dvec3 ground(0.0);
            for (int k = 0; k < 6; ++k) {
                ground = nac_.ImageToGround(l, s, h);
                h = HeightAt(ground);
            }
            const glm::dvec3 eye = nac_.Position(nac_.ImageTime(l));
            const glm::dvec3 lifted =
                ground + 10.0 * glm::normalize(eye - ground);
            const Path ball("/World/Marker" + std::to_string(i));
            // About ten pixels across.
            Sphere::Define(scene, ball).SetRadius(3.0);
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
                      std::abs(centre.y - s) < 1.0,
                  "a marker should land on its pixel");
        }
    }

    /// Every 8-bit number is the companding table's of its electrons,
    /// and the daylit ground sits within the ADC's range.
    void CheckReadout() {
        RenderProductSetOutputs outputs = RenderStrip(kStrips / 2);
        const Tensor electrons =
            Var(outputs, "/Render/Nac", "Electrons");
        const Tensor raw = Var(outputs, "/Render/Nac", "RawDN");
        std::int64_t within = 0;
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
            within += dn > 0 && dn < 255 ? 1 : 0;
        }
        std::printf("NAC: %lld of %lld numbers within the range\n",
                    static_cast<long long>(within),
                    static_cast<long long>(raw.Elements()));
        Check(within > raw.Elements() / 8,
              "the daylit ground should be exposed within the range");
    }

    IsdLineScanner nac_;
    CylindricalDem dtm_;
    CylindricalDem lola_;
    glm::dmat4 bodyToLocal_{1.0};
};

}  // namespace

ERNIE_REGISTER_DEMO(LrocNacDemo, "cameras", "lroc_nac",
                    "LROC's Narrow Angle Camera over Tycho's central "
                    "peak as image M1282557272RE's ISD has it, on "
                    "LROC's 2 m stereo terrain.");

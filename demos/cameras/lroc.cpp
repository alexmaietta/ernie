// SPDX-License-Identifier: BSD-3-Clause
// LROC's Wide Angle Camera over Tycho as it took a real image: each
// framelet posed by the image's ISD (ALE's from LRO's SPICE kernels),
// on LOLA's terrain, under the Sun where the ISD puts it, through the
// WAC's 604 nm band and readout as published, its 11-bit numbers
// companded to the 8 it sends; and the Narrow Angle Camera's readout
// from the same orbit. LOLA's terrain about Tycho is fetched into the
// cache on the first run (27 MB; not under --test); without it the
// ground is the bare reference sphere.
//
// References:
// - M. S. Robinson et al., "Lunar Reconnaissance Orbiter Camera
//   (LROC) Instrument Overview", Space Science Reviews 150:81-124,
//   2010: Table 2 (the NACs), Table 3 (the WAC), Table 4 (the WAC's
//   bandpasses and responsivities) and Appendix A (the NAC's
//   companding).
// - LROC EDR/CDR Data Product Software Interface Specification,
//   version 1.18, 2010: section 3.3 (LRO:LOOKUP_CONVERSION_TABLE) and
//   Appendix B (the WAC's companding table).
// - USGS Astrogeology, usgscsm's test data (lroWacPushFrame.json, an
//   ISD ALE wrote for a WAC image) and ALE,
//   https://github.com/DOI-USGS/usgscsm and
//   https://github.com/DOI-USGS/ale
// - D. E. Smith et al., "The Lunar Orbiter Laser Altimeter
//   Investigation on the Lunar Reconnaissance Orbiter Mission", Space
//   Science Reviews 150:209-241, 2010 (LOLA, whose LDEM is the
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
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kPi = std::numbers::pi;
constexpr double kDegree = kPi / 180.0;

/// Tycho's centre, planetocentric latitude and east longitude.
constexpr double kTychoLatitude = -43.31;
constexpr double kTychoLongitude = 348.78;

/// The terrain about it: cells of 100 m, 153.6 km across.
constexpr std::uint32_t kCells = 1537;
constexpr double kCellSize = 100.0;

/// The image lines rendered, centred on Tycho.
constexpr int kLines = 560;

/// The Sun's radius and the astronomical unit, metres; the total
/// solar irradiance at 1 au, W/m^2 (Kopp and Lean 2011).
constexpr double kSunRadius = 6.957e8;
constexpr double kAu = 1.495978707e11;
constexpr double kSolarConstant = 1361.0;

// ----------------------------------------------------------------
// NAC

/// NAC-L's published values (Robinson et al. 2010, Table 2; section
/// 3.1.2 for the KLI-5001G's QE): its 700 mm telescope and 7 um
/// pixels, focal ratio, gain, detector noise and full well.
constexpr double kNacFocalMm = 700.0;
constexpr double kNacPixelUm = 7.0;
constexpr double kNacFNumber = 3.577;
constexpr double kNacElectronsPerDn = 90.5;
constexpr double kNacReadNoise = 101.0;
constexpr double kNacFullWell = 334000.0;
constexpr int kNacSize = 128;

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
// WAC

/// One WAC band's published values: Table 4's band name, effective
/// wavelength, full width at half maximum and responsivity
/// coefficient ((W/m^3/sr) / (DN/s)), and Table 3's focal ratio and
/// pixel (9 um, 4 x 4 summed in the UV).
struct WacBand {
    int name;
    double effectiveNm;
    double fwhmNm;
    double responsivity;
    double fNumber;
    double pixelUm;
};

constexpr std::array<WacBand, 7> kWacBands = {{
    {320, 321.0, 32.0, 865.0, 5.65, 36.0},
    {360, 360.0, 15.0, 918.0, 5.65, 36.0},
    {415, 415.0, 36.0, 3727.0, 5.052, 9.0},
    {565, 566.0, 20.0, 3310.0, 5.052, 9.0},
    {605, 604.0, 20.0, 3377.0, 5.052, 9.0},
    {645, 643.0, 23.0, 3273.0, 5.052, 9.0},
    {690, 689.0, 39.0, 2612.0, 5.052, 9.0},
}};

/// The ISD's band: 604 nm.
constexpr std::size_t kIsdBand = 4;

/// The UV bands' framelets: 16 detector rows, read out 4 x 4 summed
/// (Robinson et al. 2010, section 3.2).
constexpr double kUvRows = 16.0;

/// The KAI-1001's gain, read noise and full well (Table 3).
constexpr double kWacElectronsPerDn = 25.9;
constexpr double kWacReadNoise = 66.0;
constexpr double kWacFullWell = 46100.0;

/// The WAC's 11- to 8-bit table, each 8-bit code's last 11-bit input
/// DN: the LRO:LOOKUP_CONVERSION_TABLE of its EDR labels (codes 3 and
/// 6, which no input reaches, repeat the bound before them), the same
/// as the SIS's Appendix B table of all 2048 inputs.
const std::vector<float> kWacCompand{
    1,    2,    3,    3,    4,    5,    5,    6,    7,    8,    9,
    10,   11,   13,   14,   15,   17,   18,   19,   21,   23,   24,
    26,   28,   30,   32,   33,   36,   38,   40,   42,   44,   47,
    49,   51,   54,   56,   59,   62,   64,   67,   70,   73,   76,
    79,   82,   85,   88,   92,   95,   98,   102,  105,  109,  113,
    116,  120,  124,  128,  132,  136,  140,  144,  148,  152,  156,
    161,  165,  170,  174,  179,  183,  188,  193,  198,  203,  208,
    213,  218,  223,  228,  233,  238,  244,  249,  255,  260,  266,
    271,  277,  283,  289,  295,  301,  307,  313,  319,  325,  331,
    337,  344,  350,  357,  363,  370,  376,  383,  390,  397,  404,
    411,  418,  425,  432,  439,  446,  454,  461,  468,  476,  483,
    491,  499,  506,  514,  522,  530,  538,  546,  554,  562,  570,
    579,  587,  595,  604,  612,  621,  630,  638,  647,  656,  665,
    674,  683,  692,  701,  710,  719,  728,  738,  747,  756,  766,
    776,  785,  795,  805,  814,  824,  834,  844,  854,  864,  874,
    885,  895,  905,  916,  926,  937,  947,  958,  969,  979,  990,
    1001, 1012, 1023, 1034, 1045, 1056, 1068, 1079, 1090, 1102, 1113,
    1125, 1136, 1148, 1160, 1171, 1183, 1195, 1207, 1219, 1231, 1243,
    1255, 1268, 1280, 1292, 1305, 1317, 1330, 1342, 1355, 1368, 1380,
    1393, 1406, 1419, 1432, 1445, 1458, 1472, 1485, 1498, 1512, 1525,
    1538, 1552, 1566, 1579, 1593, 1607, 1621, 1635, 1648, 1663, 1677,
    1691, 1705, 1719, 1734, 1748, 1762, 1777, 1791, 1806, 1821, 1835,
    1850, 1865, 1880, 1895, 1910, 1925, 1940, 1955, 1971, 1986, 2001,
    2017, 2032, 2047};

/// The 8-bit code of 11-bit `x` through `kWacCompand`: the first code
/// whose bound reaches it.
double WacCompand(double x) {
    const auto code =
        std::lower_bound(kWacCompand.begin(), kWacCompand.end(),
                         static_cast<float>(x));
    return std::min<double>(
        static_cast<double>(code - kWacCompand.begin()), 255.0);
}

/// The band's effective quantum efficiency: with Ernie's camera
/// equation it turns a flat spectral radiance L (W/m^3/sr) into
/// Table 4's L / RC DN per second, gain h c / (RC FWHM pi / (4 N^2) A
/// lambda), the optics', filter's and fill factor's losses all in it.
double EffectiveQe(const WacBand& band) {
    constexpr double kPlanck = 6.62607015e-34;     // J s
    constexpr double kSpeedOfLight = 299792458.0;  // m / s
    const double area = band.pixelUm * 1e-6 * band.pixelUm * 1e-6;
    const double optics = kPi / (4.0 * band.fNumber * band.fNumber);
    return kWacElectronsPerDn * kPlanck * kSpeedOfLight /
           (band.responsivity * band.fwhmNm * 1e-9 * optics * area *
            band.effectiveNm * 1e-9);
}

// ---------------------------------------------------------------
// LOLA

/// LOLA's LDEM_512 tiles at the PDS Geosciences Node: 45 degrees of
/// latitude by 90 of longitude, each 23040 lines of 46080 16-bit
/// samples, 512 to the degree.
constexpr const char* kLdemRoot =
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/lola_gdr/cylindrical/img/";
constexpr std::uint64_t kTileSamples = 46080;
constexpr int kPixelsPerDegree = 512;

/// The window of LDEM_512 about Tycho and the WAC's image over it:
/// latitudes 40 to 47 S, longitudes 345 to 352.5 E.
constexpr double kWindowNorth = -40.0;
constexpr double kWindowSouth = -47.0;
constexpr double kWindowWest = 345.0;
constexpr double kWindowEast = 352.5;

/// The cached window's label.
std::filesystem::path LdemLabel() {
    return demos::CacheDir() / "lola" / "ldem_512_lroc.lbl";
}

/// Fetches the window from LDEM_512's two tiles south of the equator
/// between 270 and 360 E, 256 lines a request, each line's samples in
/// the window alone (HTTP range requests; a tile is 2.1 GB), and
/// writes it to the cache as an image and a PDS3 label of its own,
/// which `ReadCylindricalDem` reads as it would the whole tile. As
/// demos/cameras/lroc.py's `fetch_ldem_window`.
void FetchLdemWindow() {
    const auto first = static_cast<std::uint64_t>(
        std::lround((kWindowWest - 270.0) * kPixelsPerDegree));
    const auto samples = static_cast<std::uint64_t>(
        std::lround((kWindowEast - kWindowWest) * kPixelsPerDegree));
    std::string rows;
    std::uint64_t lines = 0;
    for (const int tileNorth : {0, -45}) {
        const double top =
            std::max(std::min(kWindowNorth, double(tileNorth)),
                     tileNorth - 45.0);
        const double bottom =
            std::max(kWindowSouth, tileNorth - 45.0);
        if (top <= bottom) {
            continue;
        }
        char name[64];
        std::snprintf(name, sizeof(name),
                      "ldem_512_%02ds_%02ds_270_360", 45 - tileNorth,
                      -tileNorth);
        const std::string url =
            std::string(kLdemRoot) + name + ".img";
        const auto line0 = static_cast<std::uint64_t>(
            std::lround((tileNorth - top) * kPixelsPerDegree));
        const auto count = static_cast<std::uint64_t>(
            std::lround((top - bottom) * kPixelsPerDegree));
        std::printf("Fetching %llu lines of %llu samples from %s\n",
                    static_cast<unsigned long long>(count),
                    static_cast<unsigned long long>(samples),
                    url.c_str());
        const std::uint64_t rowBytes = 2 * kTileSamples;
        for (std::uint64_t start = line0; start < line0 + count;
             start += 256) {
            std::vector<std::pair<std::uint64_t, std::uint64_t>>
                ranges;
            for (std::uint64_t line = start;
                 line < std::min(start + 256, line0 + count);
                 ++line) {
                ranges.emplace_back(line * rowBytes + 2 * first,
                                    line * rowBytes +
                                        2 * (first + samples) - 1);
            }
            for (const std::string& part :
                 demos::FetchRanges(url, ranges)) {
                rows += part;
            }
            lines += ranges.size();
        }
    }
    const std::filesystem::path label = LdemLabel();
    std::filesystem::create_directories(label.parent_path());
    {
        std::ofstream image(label.parent_path() / "ldem_512_lroc.img",
                            std::ios::binary);
        image.write(rows.data(),
                    static_cast<std::streamsize>(rows.size()));
    }
    std::filesystem::path part = label;
    part += ".part";
    {
        std::ofstream text(part);
        text << "PDS_VERSION_ID = PDS3\n"
                "/* A window of LOLA's LDEM_512 "
                "(LRO-L-LOLA-4-GDR-V1.0), "
                "fetched by\n   demos/cameras/lroc from the PDS "
                "Geosciences Node. */\n"
                "RECORD_TYPE = FIXED_LENGTH\n"
             << "RECORD_BYTES = " << 2 * samples << "\n"
             << "FILE_RECORDS = " << lines << "\n"
             << "^IMAGE = \"LDEM_512_LROC.IMG\"\n"
                "OBJECT = IMAGE\n"
             << "  LINES = " << lines << "\n"
             << "  LINE_SAMPLES = " << samples << "\n"
             << "  SAMPLE_TYPE = LSB_INTEGER\n"
                "  SAMPLE_BITS = 16\n"
                "  UNIT = METER\n"
                "  SCALING_FACTOR = 0.5\n"
                "  OFFSET = 1737400.\n"
                "END_OBJECT = IMAGE\n"
                "OBJECT = IMAGE_MAP_PROJECTION\n"
                "  MAP_PROJECTION_TYPE = \"SIMPLE CYLINDRICAL\"\n"
             << "  MAP_RESOLUTION = " << kPixelsPerDegree
             << " <pix/deg>\n"
                "  A_AXIS_RADIUS = 1737.4 <km>\n"
             << "  MAXIMUM_LATITUDE = " << kWindowNorth << " <deg>\n"
             << "  MINIMUM_LATITUDE = "
             << kWindowNorth - double(lines) / kPixelsPerDegree
             << " <deg>\n"
             << "  WESTERNMOST_LONGITUDE = " << kWindowWest
             << " <deg>\n"
             << "  EASTERNMOST_LONGITUDE = "
             << kWindowWest + double(samples) / kPixelsPerDegree
             << " <deg>\n"
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

class LrocDemo : public app::Demo {
  public:
    LrocDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options),
          wac_(ReadIsdFile(
              demos::DataFile("lroWacPushFrame.json").string())) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // LOLA's terrain about Tycho, else the bare sphere, in a
        // frame there: x east, y north, z up. Fetched once, but not
        // for the checks, which run offline.
        const std::filesystem::path label = LdemLabel();
        if (!std::filesystem::exists(label) && !options.test &&
            demos::CanDownload()) {
            try {
                FetchLdemWindow();
            } catch (const std::exception& error) {
                std::printf("%s\n", error.what());
            }
        }
        CylindricalDem dem;
        dem.referenceRadius = 1737400.0;
        if (std::filesystem::exists(label)) {
            dem = ReadCylindricalDem(label.string());
        } else {
            std::printf("No LOLA terrain in %s; the bare sphere "
                        "stands in.\n",
                        label.parent_path().string().c_str());
        }
        dem_ = dem;
        const LocalTerrain terrain = LocalTerrainOf(
            dem, kTychoLatitude, kTychoLongitude, kCells, kCellSize);
        bodyToLocal_ = terrain.bodyToLocal;
        // The mare and highland regolith's normal albedo, about 0.1,
        // as a flat spectrum (its reddening left out): an RGB grey's
        // sigmoid spectrum dips in the UV.
        Material::Define(scene, "/World/Looks/Regolith")
            .SetReflectance(
                std::vector<float>{250.0f, 0.12f, 800.0f, 0.12f});
        Terrain::Define(scene, "/World/Terrain")
            .SetHeights(terrain.heights)
            .SetOrigin(terrain.origin)
            .SetCellSize(terrain.cellSize)
            .SetMaterialBinding(Path("/World/Looks/Regolith"));

        // The image line over Tycho, and the Sun when it was taken.
        const glm::dvec3 tycho =
            dem.referenceRadius *
            BodyDirection(kTychoLatitude, kTychoLongitude);
        const glm::dvec2 image = wac_.GroundToImage(tycho);
        firstLine_ =
            std::max(0, static_cast<int>(image.x) - kLines / 2);
        const double taken = wac_.ImageTime(image.x);
        const glm::dvec3 toSun = wac_.SunPosition(taken) - tycho;
        const double distance = glm::length(toSun);
        const glm::dvec3 local =
            glm::dmat3(bodyToLocal_) * (toSun / distance);
        std::printf("Tycho at image line %.0f; the Sun %.4f au away, "
                    "%.1f degrees up\n",
                    image.x, distance / kAu,
                    std::asin(local.z) / kDegree);
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(kSolarConstant * (kAu / distance) *
                           (kAu / distance))
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(2.0 * std::asin(kSunRadius / distance) /
                      kDegree);
        scene.Set("/World/Sun", tokens::kTransform,
                  DistantLightTransform(local));

        // The WAC, a framelet at a time, and the NAC from LRO where
        // the WAC saw Tycho, looking down at it.
        Camera::Define(scene, "/World/Wac");
        DefineProduct(
            "/Render/Wac", "/World/Wac",
            glm::dvec2(wac_.Isd().samples, wac_.FrameletLines()),
            false);
        UseWacBand(kIsdBand);
        const glm::dvec3 lro = glm::dvec3(
            bodyToLocal_ * glm::dvec4(wac_.Position(taken), 1.0));
        Camera::Define(scene, "/World/Nac");
        scene.Set(
            "/World/Nac", tokens::kTransform,
            LookAtTransform(lro, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
        DefineProduct("/Render/Nac", "/World/Nac",
                      glm::dvec2(kNacSize, kNacSize), true);
        UseNac();
        // The viewer's own view: Tycho from the south-west, 120 km
        // off and 60 km up.
        Camera::Define(scene, "/World/View")
            .SetFocalLength(35.0)
            .SetClippingRange(glm::dvec2(100.0, 1e6))
            .SetExposureTime(1.0 / 500.0);
        scene.Set("/World/View", tokens::kTransform,
                  LookAtTransform({-85e3, -85e3, 60e3},
                                  {0.0, 0.0, -2e3}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/View"))
            .SetResolution(glm::dvec2(480, 320))
            .SetAmbient(0.0);
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/View");
        }
    }

    /// The WAC's 8-bit image over Tycho, as it sends it.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, Strip());
    }

    void TestFinal() override {
        Scene& scene = GetScene();
        Camera(scene, "/World/Wac").SetSensorNoise(false);
        Camera(scene, "/World/Nac").SetSensorNoise(false);
        CheckFrameletsSeeTheGroundWhereTheIsdPutsIt();
        CheckNacCompanding();
        CheckWacCompanding();
        CheckWacResponsivities();
        Camera(scene, "/World/Wac").SetSensorNoise(true);
        Camera(scene, "/World/Nac").SetSensorNoise(true);
    }

  private:
    /// RenderProduct `product` of `camera` at `resolution`, holding
    /// RawDN, Electrons and PrimId, and LdrColor when `photographic`.
    void DefineProduct(const char* product, const char* camera,
                       glm::dvec2 resolution, bool photographic) {
        Scene& scene = GetScene();
        std::vector<std::string> names = {"RawDN", "Electrons",
                                          "PrimId"};
        if (photographic) {
            names.emplace_back("LdrColor");
        }
        std::vector<Path> vars;
        for (const std::string& var : names) {
            const Path path = Path(product).Child(var);
            RenderVar::Define(scene, path).SetSourceName(var);
            vars.push_back(path);
        }
        RenderProduct::Define(scene, product)
            .SetCamera(Path(camera))
            .SetResolution(resolution)
            .SetOrderedVars(vars);
    }

    /// The sensor filter chain of `types` under `product`, as
    /// SensorFilter prims Step0, Step1, ...; returns their paths.
    std::vector<Path> Chain(const Path& product,
                            const std::vector<std::string>& types) {
        Scene& scene = GetScene();
        std::vector<Path> filters;
        for (std::size_t i = 0; i < types.size(); ++i) {
            const Path path = product.Child("Chain").Child(
                "Step" + std::to_string(i));
            if (scene.Has(path)) {
                scene.Remove(path);
            }
            SensorFilterPrim::Define(scene, path)
                .SetFilterType(types[i]);
            filters.push_back(path);
        }
        RenderProduct(scene, product).SetFilters(filters);
        return filters;
    }

    /// The NAC's sensor and readout: electrons, their noise, the
    /// 12-bit ADC, the even and odd channels (sections 3.1.2 and
    /// 4.1.1), the 12- to 8-bit table (compandTable's defaults,
    /// Appendix A's code 0), and photographic outputs to show.
    void UseNac() {
        Scene& scene = GetScene();
        // Section 3.1.2: the KLI-5001G's average QE through the
        // 400-760 nm response is > 60%.
        Camera(scene, "/World/Nac")
            .SetFocalLength(kNacFocalMm)
            .SetHorizontalAperture(1e-3 * kNacPixelUm * kNacSize)
            .SetVerticalAperture(1e-3 * kNacPixelUm * kNacSize)
            .SetClippingRange(glm::dvec2(1e3, 1e6))
            .SetSensorType("mono")
            .SetSensorQeMono(std::vector<float>{390.0f, 0.0f, 400.0f,
                                                0.6f, 760.0f, 0.6f,
                                                770.0f, 0.0f})
            .SetExposureFStop(kNacFNumber)
            .SetExposureTime(0.002)
            .SetExposureIso(100.0)  // the gain is the ADC's
            .SetSensorElectronsPerDn(kNacElectronsPerDn)
            .SetSensorReadNoise(kNacReadNoise)
            .SetSensorFullWell(kNacFullWell)
            .SetSensorAdcBits(12)
            .SetSensorBlackLevel(0.0)
            .SetSensorNoise(true);
        const std::vector<Path> filters =
            Chain("/Render/Nac", {"qe", "noise", "adc", "readoutTaps",
                                  "compandTable", "cie", "exposure"});
        scene.Set(filters[3], "inputs:taps", 2.0);
    }

    /// Camera /World/Wac's sensor and readout through WAC band
    /// `index` (its geometry is each framelet's, from the ISD): the
    /// passband Table 4's FWHM about the effective wavelength, its
    /// edges ramped over 2 nm about the half maxima so its area is
    /// the FWHM's, at the band's effective QE (the lens's
    /// transmission in it); then electrons, their noise, the 11-bit
    /// ADC and the stored table to 8 bits.
    void UseWacBand(std::size_t index) {
        const WacBand& band = kWacBands[index];
        const double qe = EffectiveQe(band);
        const double low = band.effectiveNm - 0.5 * band.fwhmNm;
        const double high = band.effectiveNm + 0.5 * band.fwhmNm;
        Scene& scene = GetScene();
        Camera(scene, "/World/Wac")
            .SetSensorType("mono")
            .SetSensorQeMono(std::vector<float>{
                static_cast<float>(low - 1.0), 0.0f,
                static_cast<float>(low + 1.0), static_cast<float>(qe),
                static_cast<float>(high - 1.0),
                static_cast<float>(qe),
                static_cast<float>(high + 1.0), 0.0f})
            .SetSpectralRange(glm::dvec2(low - 1.0, high + 1.0))
            .SetSpectralBands(4)
            .SetLensTransmission(1.0)
            .SetExposureFStop(band.fNumber)
            .SetExposureIso(100.0)  // the gain is the ADC's
            .SetSensorElectronsPerDn(kWacElectronsPerDn)
            .SetSensorReadNoise(kWacReadNoise)
            .SetSensorFullWell(kWacFullWell)
            .SetSensorAdcBits(11)
            .SetSensorBlackLevel(0.0)
            .SetSensorNoise(true);
        const std::vector<Path> filters = Chain(
            "/Render/Wac", {"qe", "noise", "adc", "compandLookup"});
        scene.Set(filters[3], "inputs:codes", 256.0);
        scene.Set(filters[3], "inputs:uppers", kWacCompand);
    }

    /// Camera /World/Wac as framelet `framelet`, posed in the local
    /// frame; returns it.
    IsdPushFrame::Framelet UseFramelet(std::uint32_t framelet) {
        IsdPushFrame::Framelet f = wac_.FrameletOf(framelet);
        CameraParams camera = f.camera;
        camera.transform = bodyToLocal_ * camera.transform;
        AuthorIsdCamera(GetScene(), "/World/Wac", camera);
        return f;
    }

    /// The framelet holding image line `line`, in image order.
    std::uint32_t FrameletAt(int line) const {
        return static_cast<std::uint32_t>(line) /
               wac_.FrameletLines();
    }

    /// The WAC's 8-bit image of lines [firstLine_, firstLine_ +
    /// kLines), uint8 (rows, samples, 4) grey: each framelet rendered
    /// and stacked where the ISD says, its rows flipped if they run
    /// up.
    Tensor Strip() {
        const std::uint32_t samples = wac_.Isd().samples;
        const std::uint32_t height = wac_.FrameletLines();
        std::vector<std::uint8_t> pixels(
            std::size_t(kLines) * samples * 4, 0);
        const Path products[] = {"/Render/Wac"};
        for (std::uint32_t f = FrameletAt(firstLine_);
             f <= FrameletAt(firstLine_ + kLines - 1); ++f) {
            const IsdPushFrame::Framelet framelet = UseFramelet(f);
            RenderProductSetOutputs outputs =
                GetRenderer().Step(GetScene(), products);
            const Tensor raw = Var(outputs, "/Render/Wac", "RawDN");
            // A mono sensor's numbers are each pixel's first channel.
            const std::int64_t channels =
                raw.Elements() / (std::int64_t(height) * samples);
            for (std::uint32_t r = 0; r < height; ++r) {
                const int line =
                    static_cast<int>(framelet.firstLine) +
                    static_cast<int>(r) - firstLine_;
                if (line < 0 || line >= kLines) {
                    continue;
                }
                const std::uint32_t row =
                    framelet.rowsReversed ? height - 1 - r : r;
                for (std::uint32_t s = 0; s < samples; ++s) {
                    const auto dn = static_cast<std::uint8_t>(
                        raw.As<std::uint16_t>()[(row * samples + s) *
                                                channels]);
                    std::uint8_t* texel =
                        &pixels[(std::size_t(line) * samples + s) *
                                4];
                    texel[0] = texel[1] = texel[2] = dn;
                    texel[3] = 255;
                }
            }
        }
        return Tensor::Copy<std::uint8_t>(
            pixels, {kLines, static_cast<std::int64_t>(samples), 4});
    }

    /// Small balls on the ground along three pixels' rays (lifted
    /// 200 m up them, clear of the terrain's cells) show on those
    /// pixels of their framelets: the ISD's geometry, as Ernie draws
    /// it in the local frame.
    void CheckFrameletsSeeTheGroundWhereTheIsdPutsIt() {
        Scene& scene = GetScene();
        const std::array<std::array<int, 2>, 3> pixels = {
            {{firstLine_ + 100, 200},
             {firstLine_ + kLines / 2, 512},
             {firstLine_ + kLines - 100, 820}}};
        const Path products[] = {"/Render/Wac"};
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            const auto [line, sample] = pixels[i];
            const double l = line + 0.5;
            const double s = sample + 0.5;
            // On the terrain: the ray's point at the DEM's height
            // there, settled in a few rounds.
            double h = 0.0;
            glm::dvec3 ground(0.0);
            for (int k = 0; k < 4; ++k) {
                ground = wac_.ImageToGround(l, s, h);
                const glm::dvec3 d = glm::normalize(ground);
                h = dem_.HeightAt(std::asin(d.z) / kDegree,
                                  std::atan2(d.y, d.x) / kDegree);
            }
            const IsdPushFrame::Framelet framelet =
                UseFramelet(FrameletAt(line));
            const glm::dvec3 eye(framelet.camera.transform[3]);
            const glm::dvec3 at =
                glm::dvec3(bodyToLocal_ * glm::dvec4(ground, 1.0));
            const glm::dvec3 up =
                at + 200.0 * glm::normalize(
                                 glm::dvec3(bodyToLocal_ *
                                            glm::dvec4(eye, 1.0)) -
                                 at);
            const Path ball("/World/Marker" + std::to_string(i));
            // About 0.8 of a 75 m pixel across.
            Sphere::Define(scene, ball).SetRadius(30.0);
            scene.Set(ball, tokens::kTransform,
                      glm::translate(glm::dmat4(1.0), up));
            RenderProductSetOutputs outputs =
                GetRenderer().Step(scene, products);
            const Tensor ids = Var(outputs, "/Render/Wac", "PrimId");
            const std::uint32_t height = wac_.FrameletLines();
            const std::uint32_t samples = wac_.Isd().samples;
            glm::dvec2 sum(0.0);
            int hits = 0;
            for (std::uint32_t r = 0; r < height; ++r) {
                for (std::uint32_t c = 0; c < samples; ++c) {
                    if (GetRenderer().PrimPath(
                            ids.As<std::uint32_t>()[r * samples +
                                                    c]) == ball) {
                        const std::uint32_t row =
                            framelet.rowsReversed ? height - 1 - r
                                                  : r;
                        sum += glm::dvec2(row + 0.5, c + 0.5);
                        ++hits;
                    }
                }
            }
            scene.Remove(ball);
            Check(hits > 0,
                  "each marker should show in its framelet");
            const glm::dvec2 centre = sum / double(hits);
            const double row = l - framelet.firstLine;
            std::printf("marker %zu: (%.2f, %.2f) for (%.2f, %.2f)\n",
                        i, centre.x, centre.y, row, s);
            Check(std::abs(centre.x - row) < 1.5 &&
                      std::abs(centre.y - s) < 1.5,
                  "a marker should land on its pixel");
        }
    }

    /// Every NAC pixel's 8-bit number is the table's of its 12-bit
    /// one, its electrons over the gain.
    void CheckNacCompanding() {
        const Path products[] = {"/Render/Nac"};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        const Tensor electrons =
            Var(outputs, "/Render/Nac", "Electrons");
        const Tensor raw = Var(outputs, "/Render/Nac", "RawDN");
        int lit = 0;
        for (std::int64_t i = 0; i < raw.Elements(); ++i) {
            const double e = std::min<double>(
                electrons.As<float>()[i], kNacFullWell);
            // Either side of a half DN, as the WAC's.
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
        std::printf("NAC: %d lit pixels\n", lit);
        Check(lit > 100, "the NAC should see sunlit ground");
    }

    /// Every WAC pixel's 8-bit number, from black to saturation, is
    /// the stored table's of its 11-bit one.
    void CheckWacCompanding() {
        Scene& scene = GetScene();
        const IsdPushFrame::Framelet framelet =
            UseFramelet(FrameletAt(firstLine_ + kLines / 2));
        const Path products[] = {"/Render/Wac"};
        std::vector<bool> seen(256, false);
        for (double exposure = 1e-5; exposure < 4.0;
             exposure *= 1.5) {
            Camera(scene, "/World/Wac").SetExposureTime(exposure);
            RenderProductSetOutputs outputs =
                GetRenderer().Step(scene, products);
            const Tensor electrons =
                Var(outputs, "/Render/Wac", "Electrons");
            const Tensor raw = Var(outputs, "/Render/Wac", "RawDN");
            for (std::int64_t i = 0; i < raw.Elements(); ++i) {
                const double e = std::min<double>(
                    electrons.As<float>()[i], kWacFullWell);
                // Either side of a half DN, which the GPU's floats
                // may round either way.
                const auto code = [&](double scale) {
                    return WacCompand(std::clamp(
                        std::floor(e * scale / kWacElectronsPerDn +
                                   0.5),
                        0.0, 2047.0));
                };
                const std::uint16_t dn = raw.As<std::uint16_t>()[i];
                Check(
                    dn == code(1.0 - 1e-6) || dn == code(1.0 + 1e-6),
                    "the WAC should send its stored table's codes: " +
                        std::to_string(e) + " electrons sent " +
                        std::to_string(dn) + ", not " +
                        std::to_string(code(1.0)));
                seen[dn] = true;
            }
        }
        Camera(scene, "/World/Wac")
            .SetExposureTime(framelet.camera.exposureTime);
        const auto codes = std::count(seen.begin(), seen.end(), true);
        std::printf("WAC: %d codes seen\n", static_cast<int>(codes));
        Check(codes > 30,
              "the sweep should reach codes along the table");
        // The full well fills first, at 1780 DN of the 11-bit 2047.
        const auto saturated = static_cast<std::size_t>(WacCompand(
            std::floor(kWacFullWell / kWacElectronsPerDn + 0.5)));
        Check(seen[0] && seen[saturated],
              "the sweep should run from black to the full well");
    }

    /// Under a flat spectrum, each band's DN per second over a
    /// framelet is the same radiance over its responsivity
    /// coefficient: rate times RC alike across the seven, Table 4's
    /// values as Ernie renders them. The UV bands render as the
    /// camera reads them, 4 x 4 summed into 36 um pixels (over the
    /// visible framelet's geometry: the UV optics' own differ, but a
    /// pixel's rate depends only on its area and the f-number).
    void CheckWacResponsivities() {
        Scene& scene = GetScene();
        DistantLight sun(scene, "/World/Sun");
        sun.SetPhysicalIlluminant(std::string(tokens::kCustom))
            .SetPhysicalCustomIlluminant(
                std::vector<float>{250.0f, 1.0f, 800.0f, 1.0f});
        const std::uint32_t framelet =
            FrameletAt(firstLine_ + kLines / 2);
        constexpr double kExposure = 0.005;
        const Path products[] = {"/Render/Wac"};
        const auto samples = static_cast<double>(wac_.Isd().samples);
        std::vector<double> scaled;
        for (std::size_t b = 0; b < kWacBands.size(); ++b) {
            UseFramelet(framelet);
            UseWacBand(b);
            Camera camera(scene, "/World/Wac");
            camera.SetSensorNoise(false).SetExposureTime(kExposure);
            // The UV bands' framelets: 16 detector rows read 4 x 4
            // summed, 36 um pixels.
            const double summed =
                kWacBands[b].pixelUm / kWacBands[kIsdBand].pixelUm;
            if (summed > 1.0) {
                camera.SetVerticalAperture(
                    1e-3 * kWacBands[kIsdBand].pixelUm * kUvRows);
                RenderProduct(scene, "/Render/Wac")
                    .SetResolution(glm::dvec2(samples / summed,
                                              kUvRows / summed));
            }
            RenderProductSetOutputs outputs =
                GetRenderer().Step(scene, products);
            const Tensor electrons =
                Var(outputs, "/Render/Wac", "Electrons");
            double sum = 0.0;
            for (std::int64_t i = 0; i < electrons.Elements(); ++i) {
                sum += static_cast<double>(electrons.As<float>()[i]);
            }
            const double rate = sum / double(electrons.Elements()) /
                                kExposure / kWacElectronsPerDn;
            scaled.push_back(rate * kWacBands[b].responsivity);
            RenderProduct(scene, "/Render/Wac")
                .SetResolution(
                    glm::dvec2(samples, wac_.FrameletLines()));
            std::printf("WAC %d nm: %.1f DN/s, effective QE %.4f\n",
                        kWacBands[b].name, rate,
                        EffectiveQe(kWacBands[b]));
        }
        sun.SetPhysicalIlluminant(std::string(tokens::kSun));
        UseFramelet(framelet);
        UseWacBand(kIsdBand);
        Camera(scene, "/World/Wac").SetSensorNoise(false);
        for (const double value : scaled) {
            Check(std::abs(value / scaled.front() - 1.0) < 0.01,
                  "the WAC's bands should read Table 4's "
                  "responsivities");
        }
    }

    IsdPushFrame wac_;
    CylindricalDem dem_;
    glm::dmat4 bodyToLocal_{1.0};
    int firstLine_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(LrocDemo, "cameras", "lroc",
                    "LROC's Wide Angle Camera over Tycho as it took "
                    "a real image, and "
                    "its Narrow Angle Camera's readout.");

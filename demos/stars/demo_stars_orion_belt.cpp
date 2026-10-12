// SPDX-License-Identifier: BSD-3-Clause
// A zoom from Orion onto Alnitak in its belt, then a tour of how
// optics spread a star: diffraction, seeing, defocus, colour, glare,
// ghosts and blooming. Hipparcos's stars once fetched; offline,
// Orion's brightest among made-up ones.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "ernie/util/astronomy.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr std::size_t kWidth = 960;
constexpr std::size_t kHeight = 640;
constexpr std::size_t kPixels = kWidth * kHeight;
// An image's (rows, cols, 4) shape.
const std::vector<std::int64_t> kRgba = {kHeight, kWidth, 4};
constexpr double kPitchMm = 0.002;  // 2 um pixels
constexpr double kWide = 3.5;       // millimetres
constexpr double kBeltFocal = 30.0;
constexpr double kLong = 6000.0;
// A 167 mm aperture: f/12 up to 2000 mm, then a 3x Barlow to f/36.
constexpr double kAperture = 2000.0 / 12.0;
constexpr double kZoomSeconds = 10.0;
constexpr double kEffectSeconds = 3.0;
// Seconds keeping a star's peak, its light over its Airy disc's area,
// (f / N)^2 / N^2, as at 2000 mm f/12.
constexpr double kExposure = 0.002;
constexpr const char* kHipparcosUrl =
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-out.max=unlimited"
    "&-source=I/239/hip_main&-out=RAICRS,DEICRS,Vmag,B-V";

/// Right ascension and declination, degrees.
struct RaDec {
    double ra;
    double dec;
};
constexpr RaDec kOrion{83.5, 0.1};
constexpr RaDec kBelt{84.1, -1.15};
constexpr RaDec kAlnitak{85.1897, -1.9426};

/// Stars as rows of right ascension, declination, V and B-V.
using StarTable = std::vector<std::array<float, 4>>;

// Orion's brightest stars (J2000 degrees, V, B-V), for running
// offline.
const StarTable kOrionStars = {
    {88.7929f, 7.4071f, 0.42f, 1.85f},    // Betelgeuse
    {78.6345f, -8.2016f, 0.13f, -0.03f},  // Rigel
    {81.2828f, 6.3497f, 1.64f, -0.22f},   // Bellatrix
    {84.0534f, -1.2019f, 1.69f, -0.18f},  // Alnilam
    {85.1897f, -1.9426f, 1.77f, -0.21f},  // Alnitak
    {83.0017f, -0.2991f, 2.23f, -0.22f},  // Mintaka
    {86.9391f, -9.6696f, 2.09f, -0.17f},  // Saiph
    {83.7845f, 9.9342f, 3.39f, -0.16f},   // Meissa
    {83.8583f, -5.9099f, 2.77f, -0.24f},  // Hatysa
    {81.1192f, -2.3971f, 3.36f, -0.17f},  // Eta Orionis
    {84.6865f, -2.6000f, 3.81f, -0.24f},  // Sigma Orionis
    {83.8186f, -5.3897f, 5.13f, 0.02f},   // Theta1 Orionis C
};

/// Orion's brightest among made-up stars to V 9.
StarTable OfflineStars() {
    StarTable table = kOrionStars;
    const StarCatalogue made = SyntheticStars(40000, 7, 9.0);
    for (std::size_t i = 0; i < made.magnitude.size(); ++i) {
        table.push_back({made.rightAscension[i], made.declination[i],
                         made.magnitude[i], made.colorIndex[i]});
    }
    return table;
}

/// A VizieR table of right ascension, declination, V and B-V, rows
/// without a magnitude left out.
StarTable ReadVizier(const std::filesystem::path& path) {
    std::ifstream file(path);
    Check(file.good(), "cannot read " + path.string());
    StarTable table;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::array<std::string, 4> fields;
        std::istringstream in(line);
        for (std::string& field : fields) {
            std::getline(in, field, '\t');
        }
        try {
            const float ra = std::stof(fields[0]);
            const float dec = std::stof(fields[1]);
            const float v = std::stof(fields[2]);
            float bv = 0.65f;
            try {
                bv = std::stof(fields[3]);
            } catch (const std::exception&) {
            }
            table.push_back({ra, dec, v, bv});
        } catch (const std::exception&) {
            // Headers, units and blank magnitudes.
        }
    }
    return table;
}

glm::dvec3 Towards(RaDec at) {
    const double a = glm::radians(at.ra);
    const double d = glm::radians(at.dec);
    return {std::cos(d) * std::cos(a), std::cos(d) * std::sin(a),
            std::sin(d)};
}

/// Seeing's Gaussian sigma in pixels at focal length `focal` mm.
double SeeingSigma(double arcseconds, double focal) {
    return arcseconds / 2.3548 / 206265.0 * focal / kPitchMm;
}

struct FilterStep {
    std::string type;
    std::map<std::string, float> inputs = {};
};

/// An effect: its view, filter chain, the var shown and the camera's
/// changes.
struct Effect {
    std::string name;
    double focal;
    RaDec target;
    std::vector<FilterStep> steps;
    std::string var = "LdrColor";
    double stops = 0.0;
    bool achromat = false;
    bool lateral = false;
    double focusDistance = 0.0;
};

std::vector<Effect> Effects() {
    const FilterStep psf{"psf",
                         {{"profile", 1.0f}, {"maxRadius", 24.0f}}};
    const FilterStep cie{"cie"};
    const FilterStep exposure{"exposure"};
    const FilterStep qe{"qe"};
    const FilterStep adc{"adc"};
    return {
        {"Points, no optics", kLong, kAlnitak, {cie, exposure}},
        {"Gaussian fit to the Airy disc",
         kLong,
         kAlnitak,
         {{"psf", {{"profile", 0.0f}, {"maxRadius", 24.0f}}},
          cie,
          exposure}},
        {"Airy pattern", kLong, kAlnitak, {psf, cie, exposure}},
        {"Airy pattern, 33% central obstruction",
         kLong,
         kAlnitak,
         {{"psf",
           {{"profile", 1.0f},
            {"obstruction", 0.33f},
            {"maxRadius", 24.0f}}},
          cie,
          exposure}},
        {"Seeing, 1 arcsecond",
         kLong,
         kAlnitak,
         {{"psf",
           {{"profile", 0.0f},
            {"blur", static_cast<float>(SeeingSigma(1.0, kLong))},
            {"maxRadius", 24.0f}}},
          cie,
          exposure}},
        {.name = "Defocused",
         .focal = kLong,
         .target = kAlnitak,
         .steps = {psf,
                   {"defocus", {{"maxRadius", 16.0f}}},
                   cie,
                   exposure},
         .focusDistance = 20000.0},
        {.name = "Axial colour, an achromat",
         .focal = 1000.0,
         .target = kAlnitak,
         .steps = {psf,
                   {"defocus", {{"maxRadius", 32.0f}}},
                   cie,
                   exposure},
         .stops = 3.0,
         .achromat = true},
        {"The belt", kBeltFocal, kBelt, {psf, cie, exposure}},
        {.name = "Lateral colour",
         .focal = kBeltFocal,
         .target = kBelt,
         .steps = {psf, cie, exposure},
         .lateral = true},
        {.name = "Glare",
         .focal = kBeltFocal,
         .target = kBelt,
         .steps =
             {psf, cie, {"glare", {{"strength", 0.3f}}}, exposure},
         .stops = 4.0},
        // Ghosts of the saturated stars alone.
        {.name = "Lens flare ghosts",
         .focal = kBeltFocal,
         .target = kBelt,
         .steps = {psf,
                   qe,
                   {"flare",
                    {{"strength", 0.1f}, {"threshold", 15000.0f}}},
                   adc},
         .var = "RawDN",
         .stops = 10.0},
        {.name = "Blooming, a CCD",
         .focal = kBeltFocal,
         .target = kBelt,
         .steps = {psf, qe, {"blooming"}, adc},
         .var = "RawDN",
         .stops = 10.0},
    };
}

/// An achromat's secondary spectrum, (nm, focal shift in mm) pairs at
/// focal length `focal`: C and F focus together, violet focuses long.
std::vector<float> Achromat(double focal) {
    std::vector<float> pairs;
    for (int nm = 380; nm < 790; nm += 30) {
        const double x = (nm - 570.0) / 100.0;
        pairs.push_back(static_cast<float>(nm));
        pairs.push_back(static_cast<float>(3.5e-4 * x * x * focal));
    }
    return pairs;
}

/// Lateral colour: the image's scale by wavelength.
const std::vector<float> kLateral = {400.0f, 1.008f, 550.0f,
                                     1.0f,   700.0f, 0.995f};

const Path kProduct("/Render/Product");

class OrionBeltDemo : public app::Demo {
  public:
    OrionBeltDemo(app::Viewer* viewer,
                  const app::DemoOptions& options)
        : Demo(viewer, options), effects_(Effects()) {
        numFrames = 120;
        // The whole tour over a run without a window.
        fixedDt = TourSeconds() / static_cast<double>(numFrames);
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        StarField::Define(scene, "/World/Stars");
        SetStars(OfflineStars());
        Camera::Define(scene, "/World/Camera")
            .SetHorizontalAperture(kWidth * kPitchMm)
            .SetVerticalAperture(kHeight * kPitchMm)
            .SetExposureIso(100.0)
            .SetSpectralBands(16)
            .SetSensorNoise(false)
            .SetSensorType("rgb")
            .SetSensorAdcBits(8);
        RenderProduct::Define(scene, kProduct)
            .SetCamera(Path("/World/Camera"))
            .SetResolution(glm::dvec2(kWidth, kHeight));
        for (const char* var : {"LdrColor", "RawDN"}) {
            RenderVar::Define(scene, kProduct.Child(var))
                .SetSourceName(var);
        }
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(glm::dvec2(kWidth, kHeight))
            .SetProducts({kProduct});
        Zoom(0.0);

        std::vector<std::string> names;
        for (const Effect& effect : effects_) {
            names.push_back(effect.name);
        }
        Widgets().Checkbox("Tour", true,
                           [this](bool on) { tour_ = on; });
        Widgets().Combo("Effect", std::move(names), 0,
                        [this](std::size_t index) {
                            tour_ = false;
                            Apply(static_cast<int>(index));
                        });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }

        // Hipparcos when cached; else fetched in the background, but
        // not for the checks, which run offline.
        catalogue_ = demos::CacheDir() / "stars" / "hipparcos.tsv";
        if (!options.test) {
            if (std::filesystem::exists(catalogue_)) {
                Load();
            } else if (demos::CanDownload()) {
                download_ = std::make_unique<demos::AsyncDownload>(
                    kHipparcosUrl, catalogue_);
            }
        }
    }

    void Step() override {
        if (download_ != nullptr) {
            for (const auto& message : download_->TakeMessages()) {
                Report(message.text, message.error);
            }
            if (download_->Done()) {
                if (download_->Succeeded()) {
                    Load();
                }
                download_.reset();
            }
        }
        if (!tour_) {
            return;
        }
        const double t = std::fmod(time, TourSeconds());
        if (t < kZoomSeconds) {
            Zoom(t / kZoomSeconds);
        } else {
            const int index =
                static_cast<int>((t - kZoomSeconds) / kEffectSeconds);
            if (index != shown_) {
                Apply(index);
            }
        }
    }

    /// The effects in a grid, three to a row.
    void SaveOutput(const std::filesystem::path& path) override {
        const std::vector<Tensor> images = RenderEffects();
        std::vector<Tensor> rows;
        for (std::size_t i = 0; i < images.size(); i += 3) {
            rows.push_back(
                demos::SideBySide(std::span(images).subspan(
                    i, std::min<std::size_t>(3, images.size() - i))));
        }
        const std::int64_t width = rows[0].Shape()[1];
        std::vector<std::uint8_t> pixels;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            if (r > 0) {  // a black gap of four rows
                pixels.insert(pixels.end(),
                              static_cast<std::size_t>(16 * width),
                              0);
            }
            const auto* data =
                static_cast<const std::uint8_t*>(rows[r].Data());
            pixels.insert(pixels.end(), data,
                          data + rows[r].Elements());
        }
        const auto height =
            static_cast<std::int64_t>(pixels.size()) / (4 * width);
        app::WritePng(path, Tensor::Copy<std::uint8_t>(
                                pixels, {height, width, 4}));
    }

    void TestFinal() override {
        const std::vector<Tensor> images = RenderEffects();
        std::vector<std::vector<double>> luminance;
        for (std::size_t i = 0; i < images.size(); ++i) {
            luminance.push_back(
                Luminance(images[i], effects_[i].var == "LdrColor"));
        }
        // Each pixel's distance from the image's centre.
        const auto radius = [](std::size_t p) {
            return std::hypot(static_cast<double>(p % kWidth) -
                                  0.5 * kWidth + 0.5,
                              static_cast<double>(p / kWidth) -
                                  0.5 * kHeight + 0.5);
        };
        const auto halfLightRadius = [&](std::size_t index) {
            std::vector<std::pair<double, double>> near;
            for (std::size_t p = 0; p < kPixels; ++p) {
                if (radius(p) < 60.0) {
                    near.emplace_back(radius(p), luminance[index][p]);
                }
            }
            std::sort(near.begin(), near.end());
            double total = 0.0;
            for (const auto& [r, l] : near) {
                total += l;
            }
            double sum = 0.0;
            for (const auto& [r, l] : near) {
                sum += l;
                if (sum >= 0.5 * total) {
                    return r;
                }
            }
            return 60.0;
        };
        std::vector<double> radii;
        for (std::size_t i = 0; i < 6; ++i) {
            radii.push_back(halfLightRadius(i));
            std::printf("%s: half the light within %.1f px\n",
                        effects_[i].name.c_str(), radii.back());
        }
        Check(radii[0] < 1.5, "a bare star lands in a pixel or two");
        Check(radii[1] > radii[0] && radii[2] > radii[0],
              "diffraction spreads the star");
        // Beyond the first dark ring, 1.22 lambda N at 550 nm.
        const double ring =
            1.22 * 0.55e-3 * kLong / kAperture / kPitchMm;
        double rings[2];
        for (std::size_t k = 0; k < 2; ++k) {
            double outer = 0.0;
            double all = 0.0;
            for (std::size_t p = 0; p < kPixels; ++p) {
                const double r = radius(p);
                if (r < 60.0) {
                    all += luminance[2 + k][p];
                    outer += r > ring ? luminance[2 + k][p] : 0.0;
                }
            }
            rings[k] = outer / all;
        }
        std::printf(
            "light in the rings: clear %.3f, obstructed %.3f\n",
            rings[0], rings[1]);
        Check(rings[1] > rings[0],
              "an obstruction moves light into the rings");
        Check(radii[4] > 1.5 * radii[2],
              "seeing blurs beyond diffraction");
        Check(radii[5] > 1.5 * radii[2], "defocus spreads the star");

        // Axial colour: a halo bluer than the core.
        glm::dvec3 core(0.0);
        glm::dvec3 halo(0.0);
        for (std::size_t p = 0; p < kPixels; ++p) {
            const double r = radius(p);
            if (r < 2.0) {
                core += LinearAt(images[6], p);
            } else if (r > 5.0 && r < 14.0) {
                halo += LinearAt(images[6], p);
            }
        }
        std::printf(
            "axial colour: core %.3f %.3f %.3f, halo %.4f %.4f "
            "%.4f\n",
            core.r, core.g, core.b, halo.r, halo.g, halo.b);
        Check(halo.b / std::max(halo.g, 1e-9) >
                  core.b / std::max(core.g, 1e-9),
              "an achromat's halo is bluer than its core");

        // Lateral colour: off-axis stars' red and blue apart,
        // radially.
        const auto colourSplit = [&](const Tensor& image) {
            std::size_t brightest = 0;
            double most = -1.0;
            for (std::size_t p = 0; p < kPixels; ++p) {
                const glm::dvec3 c = LinearAt(image, p);
                if (radius(p) > 150.0 && c.r + c.g + c.b > most) {
                    most = c.r + c.g + c.b;
                    brightest = p;
                }
            }
            glm::dvec2 red(0.0);
            glm::dvec2 blue(0.0);
            double redSum = 0.0;
            double blueSum = 0.0;
            for (int dy = -8; dy <= 8; ++dy) {
                for (int dx = -8; dx <= 8; ++dx) {
                    // Wraps past the edges; the stars sit well
                    // inside.
                    const glm::dvec3 c = LinearAt(
                        image,
                        brightest +
                            static_cast<std::size_t>(
                                dy * static_cast<int>(kWidth) + dx));
                    red += c.r * glm::dvec2(dx, dy);
                    blue += c.b * glm::dvec2(dx, dy);
                    redSum += c.r;
                    blueSum += c.b;
                }
            }
            return glm::length(red / std::max(redSum, 1e-9) -
                               blue / std::max(blueSum, 1e-9));
        };
        const double plain = colourSplit(images[7]);
        const double lateral = colourSplit(images[8]);
        std::printf("red to blue off axis: %.2f px, lateral colour "
                    "%.2f px\n",
                    plain, lateral);
        Check(lateral > plain + 0.5,
              "lateral colour splits red from blue");

        const auto lit = [&](std::size_t index) {
            return std::count_if(luminance[index].begin(),
                                 luminance[index].end(),
                                 [](double l) { return l > 0.01; });
        };
        std::printf("lit pixels: belt %td, glare %td\n", lit(7),
                    lit(9));
        Check(lit(9) > 2 * lit(7), "glare spreads a halo");

        // Ghosts: light on the dark sky the same exposure lacks
        // without them.
        Apply(10);
        GetScene().Set(kProduct.Child("Chain").Child("Step2"),
                       "inputs:strength", 0.0f);
        const Tensor clean = RenderShown();
        Apply(10);  // the ghosts back
        const auto* with =
            static_cast<const std::uint8_t*>(images[10].Data());
        const auto* without =
            static_cast<const std::uint8_t*>(clean.Data());
        double ghosts = 0.0;
        for (std::size_t p = 0; p < kPixels; ++p) {
            if (luminance[7][p] < 0.002) {
                for (std::size_t c = 0; c < 3; ++c) {
                    ghosts += std::max(0, with[p * 4 + c] -
                                              without[p * 4 + c]);
                }
            }
        }
        std::printf("ghost DN on the dark sky: %.0f\n", ghosts);
        Check(ghosts > 1000.0, "flare puts ghosts on the dark sky");

        // Blooming: the brightest star streaks down its column.
        const std::vector<double>& bloom = luminance[11];
        const auto peak = static_cast<std::size_t>(
            std::max_element(bloom.begin(), bloom.end()) -
            bloom.begin());
        int tall = 0;
        int wide = 0;
        for (std::size_t y = 0; y < kHeight; ++y) {
            tall += bloom[y * kWidth + peak % kWidth] > 0.5 ? 1 : 0;
        }
        for (std::size_t x = 0; x < kWidth; ++x) {
            wide += bloom[peak / kWidth * kWidth + x] > 0.5 ? 1 : 0;
        }
        std::printf("blooming: %d px tall, %d px wide\n", tall, wide);
        Check(tall > 3 * wide, "blooming streaks along columns");
    }

  private:
    double TourSeconds() const {
        return kZoomSeconds +
               kEffectSeconds * static_cast<double>(effects_.size());
    }

    void SetStars(const StarTable& table) {
        const auto column = [&](std::size_t c) {
            std::vector<float> values;
            values.reserve(table.size());
            for (const auto& row : table) {
                values.push_back(row[c]);
            }
            return Tensor::Copy<float>(
                values, {static_cast<std::int64_t>(values.size())});
        };
        StarField(GetScene(), "/World/Stars")
            .SetRightAscension(column(0))
            .SetDeclination(column(1))
            .SetMagnitude(column(2))
            .SetColorIndex(column(3));
    }

    /// A message in the viewer's log, else printed.
    void Report(const std::string& text, bool error) {
        if (app::Viewer* viewer = GetViewer()) {
            viewer->Log(error ? app::LogLevel::Error
                              : app::LogLevel::Info,
                        text);
        } else {
            std::printf("%s\n", text.c_str());
        }
    }

    void Load() {
        try {
            const StarTable table = ReadVizier(catalogue_);
            SetStars(table);
            Report("Loaded " + std::to_string(table.size()) +
                       " Hipparcos stars",
                   false);
        } catch (const std::exception& error) {
            Report(error.what(), true);
        }
    }

    void Aim(double focal, RaDec target) {
        const double n = std::max(12.0, focal / kAperture);
        const double peak = n * n / focal / 0.072;
        Camera(GetScene(), "/World/Camera")
            .SetFocalLength(focal)
            .SetFStop(n)
            .SetExposureFStop(n)
            .SetExposureTime(kExposure * peak * peak)
            .SetTransform(LookAtTransform(
                glm::dvec3(0.0), Towards(target), {0.0, 0.0, 1.0}));
    }

    /// The product's filters and the var shown; a raw chain keeps
    /// LdrColor too, for frames rendered as photographs.
    void Chain(std::vector<FilterStep> steps,
               const std::string& var) {
        Scene& scene = GetScene();
        const Path chain = kProduct.Child("Chain");
        for (std::size_t i = 0; i < chainLength_; ++i) {
            scene.Remove(chain.Child("Step" + std::to_string(i)));
        }
        std::vector<Path> vars = {kProduct.Child("LdrColor")};
        if (var == "RawDN") {
            steps.push_back({"cie"});
            steps.push_back({"exposure"});
            vars.push_back(kProduct.Child(var));
        }
        var_ = var;
        std::vector<Path> filters;
        for (std::size_t i = 0; i < steps.size(); ++i) {
            const Path path = chain.Child("Step" + std::to_string(i));
            SensorFilterPrim::Define(scene, path)
                .SetFilterType(steps[i].type);
            for (const auto& [name, value] : steps[i].inputs) {
                scene.Set(path, "inputs:" + name, value);
            }
            filters.push_back(path);
        }
        chainLength_ = steps.size();
        RenderProduct(scene, kProduct)
            .SetFilters(filters)
            .SetOrderedVars(vars);
        if (app::Viewer* viewer = GetViewer()) {
            viewer->SetViewVar(var);
        }
    }

    void Lens(double focal, bool achromat, bool lateral,
              double focus) {
        Camera(GetScene(), "/World/Camera")
            .SetFocusDistance(focus)
            .SetLensChromaticFocus(
                achromat ? Achromat(focal)
                         : std::vector<float>{550.0f, 0.0f})
            .SetLensChromaticScale(
                lateral ? kLateral
                        : std::vector<float>{550.0f, 1.0f});
    }

    /// The zoom at `u` of the way: focal length growing
    /// exponentially, the aim moving from Orion to the belt to
    /// Alnitak.
    void Zoom(double u) {
        if (shown_ != kZooming) {
            Chain(
                {{"psf", {{"profile", 1.0f}}}, {"cie"}, {"exposure"}},
                "LdrColor");
            Lens(kWide, false, false, 0.0);
            shown_ = kZooming;
        }
        const double e = u * u * (3.0 - 2.0 * u);
        const double focal = kWide * std::pow(kLong / kWide, e);
        const double k =
            std::log(focal / kWide) / std::log(kBeltFocal / kWide);
        RaDec a = kOrion;
        RaDec b = kBelt;
        double w = k;
        if (k > 1.0) {
            a = kBelt;
            b = kAlnitak;
            w = std::min(std::log(focal / kBeltFocal) /
                             std::log(400.0 / kBeltFocal),
                         1.0);
        }
        const glm::dvec3 d =
            glm::normalize((1.0 - w) * Towards(a) + w * Towards(b));
        Aim(focal, {glm::degrees(std::atan2(d.y, d.x)),
                    glm::degrees(std::asin(d.z))});
    }

    /// Effect `index`.
    void Apply(int index) {
        const Effect& effect =
            effects_.at(static_cast<std::size_t>(index));
        Aim(effect.focal, effect.target);
        Camera camera(GetScene(), "/World/Camera");
        camera.SetExposureTime(camera.GetExposureTime() *
                               std::exp2(effect.stops));
        Lens(effect.focal, effect.achromat, effect.lateral,
             effect.focusDistance);
        Chain(effect.steps, effect.var);
        shown_ = index;
    }

    /// What the product shows now as uint8 RGBA: LdrColor, or the RGB
    /// sensor's 8-bit numbers.
    Tensor RenderShown() {
        const Path products[] = {kProduct};
        RenderProductSetOutputs outputs =
            GetRenderer().Step(GetScene(), products);
        const Tensor values = outputs.at(kProduct)
                                  .vars.at(var_)
                                  .Map(MapDevice::Cpu)
                                  .tensor;
        if (values.Type() == DType::UInt8) {
            const auto* data =
                static_cast<const std::uint8_t*>(values.Data());
            return Tensor::Copy<std::uint8_t>(
                std::vector<std::uint8_t>(data, data + kPixels * 4),
                kRgba);
        }
        const auto channels =
            static_cast<std::size_t>(values.Elements()) / kPixels;
        const auto dn = values.As<std::uint16_t>();
        std::vector<std::uint8_t> pixels(kPixels * 4, 255);
        for (std::size_t p = 0; p < kPixels; ++p) {
            for (std::size_t c = 0; c < 3; ++c) {
                pixels[p * 4 + c] = static_cast<std::uint8_t>(
                    std::min<int>(dn[p * channels + c], 255));
            }
        }
        return Tensor::Copy<std::uint8_t>(pixels, kRgba);
    }

    /// Every effect's image; leaves the demo as it was.
    std::vector<Tensor> RenderEffects() {
        const int shown = shown_;
        std::vector<Tensor> images;
        for (std::size_t i = 0; i < effects_.size(); ++i) {
            Apply(static_cast<int>(i));
            images.push_back(RenderShown());
            LogImage("effects/" + std::to_string(i), images.back());
        }
        if (shown >= 0) {
            Apply(shown);
        }
        return images;
    }

    /// Pixel `p`'s linear colour.
    static glm::dvec3 LinearAt(const Tensor& image, std::size_t p) {
        const auto* px =
            static_cast<const std::uint8_t*>(image.Data()) + p * 4;
        return demos::SrgbToLinear(glm::dvec3(px[0], px[1], px[2]) /
                                   255.0);
    }

    /// Each pixel's luminance: decoded from sRGB, or the mean of the
    /// sensor's numbers over 255.
    static std::vector<double> Luminance(const Tensor& image,
                                         bool srgb) {
        std::vector<double> out(kPixels);
        const auto* px =
            static_cast<const std::uint8_t*>(image.Data());
        for (std::size_t p = 0; p < kPixels; ++p) {
            if (srgb) {
                out[p] = glm::dot(LinearAt(image, p),
                                  glm::dvec3(0.2126, 0.7152, 0.0722));
            } else {
                out[p] = (px[p * 4] + px[p * 4 + 1] + px[p * 4 + 2]) /
                         (3.0 * 255.0);
            }
        }
        return out;
    }

    static constexpr int kZooming = -1;

    std::vector<Effect> effects_;
    std::filesystem::path catalogue_;
    std::unique_ptr<demos::AsyncDownload> download_;
    std::size_t chainLength_ = 0;
    std::string var_;  ///< The var shown.
    int shown_ = -2;
    bool tour_ = true;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    OrionBeltDemo, "stars", "orion_belt",
    "A zoom from Orion onto Alnitak, then how optics spread a star: "
    "the Airy pattern, an obstruction, seeing, defocus, axial and "
    "lateral colour, glare, flare ghosts and blooming.");

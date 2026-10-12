// SPDX-License-Identifier: BSD-3-Clause
// Orion from space through four star catalogues: made-up stars, the
// Yale Bright Star Catalogue (to V 6.5), Hipparcos (to about V 12)
// and Tycho-2 around Orion (to VT 11), fetched from VizieR. Offline,
// made-up stars to each one's depth around Orion's brightest.

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
constexpr double kOrionRa = 83.8;  // degrees
constexpr double kOrionDec = -1.2;
const std::string kVizier = "https://vizier.cds.unistra.fr/viz-bin/"
                            "asu-tsv?-out.max=unlimited&";

/// A catalogue: its cached file (none for made-up stars) and url, and
/// offline made-up stars' count and faintest magnitude. Tycho-2 rows
/// give BT and VT.
struct Catalogue {
    std::string name;
    std::string file;
    std::string url;
    std::size_t count;
    double faintest;
};

const std::array<Catalogue, 4> kCatalogues = {{
    {"Made-up stars", "", "", 9110, 6.5},
    {"Yale Bright Star Catalogue", "bsc5.tsv",
     kVizier + "-source=V/50/catalog&-out=_RAJ2000,_DEJ2000,Vmag,B-V",
     9110, 6.5},
    {"Hipparcos", "hipparcos.tsv",
     kVizier + "-source=I/239/hip_main&-out=RAICRS,DEICRS,Vmag,B-V",
     118000, 9.5},
    {"Tycho-2 around Orion", "tycho2_orion.tsv",
     kVizier + "-source=I/259/tyc2&-out=RAmdeg,DEmdeg,BTmag,VTmag"
               "&-c=83.8,-1.2&-c.rd=25&VTmag=%3C11",
     400000, 11.0},
}};

/// Stars as rows of right ascension, declination, V and B-V.
using StarTable = std::vector<std::array<float, 4>>;

// Orion's brightest stars and neighbours (J2000 degrees, V, B-V).
const StarTable kBrightStars = {
    {88.7929f, 7.4071f, 0.42f, 1.85f},      // Betelgeuse
    {78.6345f, -8.2016f, 0.13f, -0.03f},    // Rigel
    {81.2828f, 6.3497f, 1.64f, -0.22f},     // Bellatrix
    {84.0534f, -1.2019f, 1.69f, -0.18f},    // Alnilam
    {85.1897f, -1.9426f, 1.77f, -0.21f},    // Alnitak
    {83.0017f, -0.2991f, 2.23f, -0.22f},    // Mintaka
    {86.9391f, -9.6696f, 2.09f, -0.17f},    // Saiph
    {83.7845f, 9.9342f, 3.39f, -0.16f},     // Meissa
    {83.8583f, -5.9099f, 2.77f, -0.24f},    // Hatysa
    {101.2872f, -16.7161f, -1.46f, 0.00f},  // Sirius
    {114.8255f, 5.2250f, 0.34f, 0.42f},     // Procyon
    {68.9802f, 16.5093f, 0.87f, 1.54f},     // Aldebaran
};

StarTable MadeUp(std::size_t count, double faintest,
                 std::uint64_t seed) {
    const StarCatalogue made = SyntheticStars(count, seed, faintest);
    StarTable table;
    for (std::size_t i = 0; i < made.magnitude.size(); ++i) {
        table.push_back({made.rightAscension[i], made.declination[i],
                         made.magnitude[i], made.colorIndex[i]});
    }
    return table;
}

/// A VizieR table of right ascension, declination, V and B-V, rows
/// without a magnitude left out. Tycho-2's BT and VT become V = VT -
/// 0.090 (BT - VT) and B - V = 0.850 (BT - VT) (ESA SP-1200, 1997,
/// volume 1, section 1.3).
StarTable ReadVizier(const std::filesystem::path& path) {
    std::ifstream file(path);
    Check(file.good(), "cannot read " + path.string());
    StarTable table;
    bool tycho = false;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        tycho = tycho || line.find("BTmag") != std::string::npos;
        std::array<std::string, 4> fields;
        std::istringstream in(line);
        for (std::string& field : fields) {
            std::getline(in, field, '\t');
        }
        const auto number = [](const std::string& text) {
            try {
                return std::stod(text);
            } catch (const std::exception&) {
                return std::nan("");
            }
        };
        const double ra = number(fields[0]);
        const double dec = number(fields[1]);
        double a = number(fields[2]);
        double b = number(fields[3]);
        if (std::isnan(ra) || std::isnan(dec)) {
            continue;  // headers and units
        }
        if (tycho) {
            const double bt = a;
            const double vt = b;
            a = std::isnan(bt) ? vt : vt - 0.090 * (bt - vt);
            b = std::isnan(bt) ? 0.65 : 0.850 * (bt - vt);
        }
        if (!std::isnan(a)) {
            table.push_back(
                {static_cast<float>(ra), static_cast<float>(dec),
                 static_cast<float>(a),
                 static_cast<float>(std::isnan(b) ? 0.65 : b)});
        }
    }
    return table;
}

/// The stars seen in a uint8 RGBA image: pixels no dimmer than their
/// eight neighbours and brighter by `threshold` (linear) than any
/// three pixels away.
std::size_t CountStars(const Tensor& image,
                       double threshold = 0.003) {
    const auto* px = static_cast<const std::uint8_t*>(image.Data());
    std::vector<double> lum(kWidth * kHeight);
    for (std::size_t p = 0; p < lum.size(); ++p) {
        lum[p] = glm::dot(
            demos::SrgbToLinear(
                glm::dvec3(px[p * 4], px[p * 4 + 1], px[p * 4 + 2]) /
                255.0),
            glm::dvec3(0.2126, 0.7152, 0.0722));
    }
    std::size_t count = 0;
    for (std::size_t y = 3; y + 3 < kHeight; ++y) {
        for (std::size_t x = 3; x + 3 < kWidth; ++x) {
            const double here = lum[y * kWidth + x];
            bool peak = true;
            double ring = 0.0;
            for (std::size_t v = y - 3; v <= y + 3; ++v) {
                for (std::size_t u = x - 3; u <= x + 3; ++u) {
                    const std::size_t d = std::max(
                        v > y ? v - y : y - v, u > x ? u - x : x - u);
                    const double there = lum[v * kWidth + u];
                    if (d == 1) {
                        peak = peak && here >= there;
                    } else if (d == 3) {
                        ring = std::max(ring, there);
                    }
                }
            }
            count += peak && here > ring + threshold ? 1 : 0;
        }
    }
    return count;
}

Path FieldOf(std::size_t index) {
    return Path("/World/Catalogue" + std::to_string(index));
}

class CataloguesDemo : public app::Demo {
  public:
    CataloguesDemo(app::Viewer* viewer,
                   const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        for (std::size_t i = 0; i < kCatalogues.size(); ++i) {
            StarField::Define(scene, FieldOf(i));
            StarTable stars = i > 0 ? kBrightStars : StarTable{};
            const StarTable made = MadeUp(
                kCatalogues[i].count, kCatalogues[i].faintest, i + 1);
            stars.insert(stars.end(), made.begin(), made.end());
            SetStars(i, stars);
        }
        const double ra = glm::radians(kOrionRa);
        const double dec = glm::radians(kOrionDec);
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(50.0)
            .SetHorizontalAperture(36.0)
            .SetVerticalAperture(24.0)
            .SetExposureFStop(2.0)
            .SetExposureTime(8.0)
            .SetExposureIso(800.0)
            .SetSensorNoise(false)
            .SetTransform(LookAtTransform(
                glm::dvec3(0.0),
                {std::cos(dec) * std::cos(ra),
                 std::cos(dec) * std::sin(ra), std::sin(dec)},
                {0.0, 0.0, 1.0}));
        const std::pair<const char*, std::map<std::string, float>>
            steps[] = {{"psf", {{"blur", 0.8f}, {"maxRadius", 4.0f}}},
                       {"cie", {}},
                       {"glare", {{"strength", 0.05f}}},
                       {"exposure", {}}};
        const Path product("/Render/Product");
        std::vector<Path> filters;
        for (std::size_t i = 0; i < std::size(steps); ++i) {
            const Path path = product.Child("Chain").Child(
                "Step" + std::to_string(i));
            SensorFilterPrim::Define(scene, path)
                .SetFilterType(steps[i].first);
            for (const auto& [name, value] : steps[i].second) {
                scene.Set(path, "inputs:" + name, value);
            }
            filters.push_back(path);
        }
        RenderProduct::Define(scene, product)
            .SetCamera(Path("/World/Camera"))
            .SetResolution(glm::dvec2(kWidth, kHeight))
            .SetFilters(filters);
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(glm::dvec2(kWidth, kHeight))
            .SetProducts({product});
        Show(1);

        std::vector<std::string> names;
        for (const Catalogue& catalogue : kCatalogues) {
            names.emplace_back(catalogue.name);
        }
        Widgets().Combo("Catalogue", std::move(names), 1,
                        [this](std::size_t index) { Show(index); });
        Widgets().Slider(
            "Magnitude limit", 0.0, 12.0, 12.0,
            [this](double magnitude) { Limit(magnitude); });
        Widgets().Slider("Exposure (stops)", -6.0, 4.0, 0.0,
                         [this](double stops) {
                             Camera(GetScene(), "/World/Camera")
                                 .SetExposure(stops);
                         });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }

        // The catalogues when cached; else fetched in the background,
        // but not for the checks, which run offline.
        if (!options.test) {
            for (std::size_t i = 0; i < kCatalogues.size(); ++i) {
                if (kCatalogues[i].file.empty()) {
                    continue;
                }
                const std::filesystem::path path =
                    demos::CacheDir() / "stars" / kCatalogues[i].file;
                if (std::filesystem::exists(path)) {
                    Load(i, path);
                } else if (demos::CanDownload()) {
                    downloads_[i] = {
                        std::make_unique<demos::AsyncDownload>(
                            kCatalogues[i].url, path),
                        path};
                }
            }
        }
    }

    void Step() override {
        for (auto it = downloads_.begin(); it != downloads_.end();) {
            auto& [download, path] = it->second;
            for (const auto& message : download->TakeMessages()) {
                Report(message.text, message.error);
            }
            if (!download->Done()) {
                ++it;
                continue;
            }
            if (download->Succeeded()) {
                Load(it->first, path);
            }
            it = downloads_.erase(it);
        }
    }

    /// The catalogues two by two.
    void SaveOutput(const std::filesystem::path& path) override {
        const std::vector<Tensor> images = RenderAll();
        const Tensor top =
            demos::SideBySide(std::span(images).first(2));
        const Tensor bottom =
            demos::SideBySide(std::span(images).last(2));
        const auto bytes = static_cast<std::size_t>(top.Elements());
        std::vector<std::uint8_t> pixels(
            2 * bytes + 4 * 4 * (2 * kWidth + 4), 0);
        const auto* a = static_cast<const std::uint8_t*>(top.Data());
        const auto* b =
            static_cast<const std::uint8_t*>(bottom.Data());
        std::copy(a, a + bytes, pixels.begin());
        std::copy(b, b + bytes,
                  pixels.end() - static_cast<std::ptrdiff_t>(bytes));
        for (std::size_t i = 3; i < pixels.size(); i += 4) {
            pixels[i] = 255;
        }
        app::WritePng(
            path, Tensor::Copy<std::uint8_t>(
                      pixels, {2 * kHeight + 4, 2 * kWidth + 4, 4}));
    }

    void TestFinal() override {
        const std::vector<Tensor> images = RenderAll();
        std::vector<std::size_t> counts;
        for (std::size_t i = 0; i < images.size(); ++i) {
            counts.push_back(CountStars(images[i]));
            std::printf("%s: %zu stars\n",
                        kCatalogues[i].name.c_str(), counts[i]);
        }
        Check(static_cast<double>(counts[2]) >
                  1.3 * static_cast<double>(counts[1]),
              "Hipparcos goes deeper");
        Check(counts[3] > counts[1], "Tycho-2 goes deeper");
        Show(2);
        Limit(4.0);
        const std::size_t bright =
            CountStars(GetRenderer().Render(GetScene()));
        Limit(12.0);
        std::printf("Hipparcos to V 4: %zu stars\n", bright);
        Check(bright > 5 && 2 * bright < counts[1],
              "the limit leaves the faint out");
    }

  private:
    void SetStars(std::size_t index, const StarTable& table) {
        const auto column = [&](std::size_t c) {
            std::vector<float> values;
            values.reserve(table.size());
            for (const auto& row : table) {
                values.push_back(row[c]);
            }
            return Tensor::Copy<float>(
                values, {static_cast<std::int64_t>(values.size())});
        };
        StarField(GetScene(), FieldOf(index))
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

    void Load(std::size_t index, const std::filesystem::path& path) {
        try {
            const StarTable table = ReadVizier(path);
            SetStars(index, table);
            Report("Loaded " + std::to_string(table.size()) +
                       " stars of " + kCatalogues[index].name,
                   false);
        } catch (const std::exception& error) {
            Report(error.what(), true);
        }
    }

    /// Catalogue `index` alone: the first visible star field shows.
    void Show(std::size_t index) {
        shown_ = index;
        for (std::size_t i = 0; i < kCatalogues.size(); ++i) {
            StarField(GetScene(), FieldOf(i))
                .SetVisibility(std::string(i == index
                                               ? tokens::kInherited
                                               : tokens::kInvisible));
        }
    }

    void Limit(double magnitude) {
        for (std::size_t i = 0; i < kCatalogues.size(); ++i) {
            StarField(GetScene(), FieldOf(i))
                .SetMagnitudeLimit(magnitude);
        }
    }

    /// Every catalogue's view; leaves the catalogue shown.
    std::vector<Tensor> RenderAll() {
        const std::size_t shown = shown_;
        std::vector<Tensor> images;
        for (std::size_t i = 0; i < kCatalogues.size(); ++i) {
            Show(i);
            images.push_back(GetRenderer().Render(GetScene()));
        }
        Show(shown);
        return images;
    }

    std::map<std::size_t,
             std::pair<std::unique_ptr<demos::AsyncDownload>,
                       std::filesystem::path>>
        downloads_;
    std::size_t shown_ = 1;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    CataloguesDemo, "stars", "catalogues",
    "Orion from space through four star catalogues: made-up stars, "
    "the "
    "Yale Bright Star Catalogue, Hipparcos and Tycho-2, fetched from "
    "VizieR.");

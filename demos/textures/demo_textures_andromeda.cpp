// SPDX-License-Identifier: BSD-3-Clause
// Hubble's panorama of the Andromeda Galaxy on a plane: up to 40000 x
// 12788 pixels (half a gigapixel), fetched once into the cache, baked
// a band at a time to BC7 on the GPU off the frame and streamed as a
// virtual texture as the camera zooms in, only the tiles in view
// resident. Image: NASA, ESA, J. Dalcanton, B. F. Williams, L. C.
// Johnson (University of Washington), the PHAT team and R. Gendler
// (ESA/Hubble heic1502a, CC BY 4.0).

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "util.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

// The plane: as wide as the panorama is long, in metres, and as tall
// as its sides' ratio.
constexpr double kWidth = 32.0;
constexpr double kHeight = kWidth * 3197.0 / 10000.0;
// The camera's distance with the whole width in view (a 24 mm lens on
// a 20.955 mm wide aperture).
constexpr double kWhole = 0.5 * kWidth * 24.0 / (0.5 * 20.955) * 1.05;

/// An image: its label, URL and cached file name.
struct Picture {
    const char* label;
    const char* url;
    const char* file;
};

// ESA/Hubble's publication TIFFs of heic1502a, 10000 and 40000 pixels
// wide.
const std::array<Picture, 2> kPictures = {{
    {"10000 x 3197 (114 MB)",
     "https://esahubble.org/media/archives/images/publicationtiff10k/"
     "heic1502a.tif",
     "heic1502a_10k.tif"},
    {"40000 x 12788 (1.7 GB)",
     "https://esahubble.org/media/archives/images/publicationtiff40k/"
     "heic1502a.tif",
     "heic1502a_40k.tif"},
}};

/// A stand-in for the checks, which run offline: 9000 x 2880 pixels
/// (wider than a virtual texture's side, so it streams as the real
/// ones do), red on the left half and blue on the right, a green
/// square 32 pixels across at three quarters of the way along.
std::filesystem::path StandIn() {
    constexpr std::int64_t kCols = 9000;
    constexpr std::int64_t kRows = 2880;
    std::vector<std::uint8_t> texels(kCols * kRows * 4);
    for (std::int64_t row = 0; row < kRows; ++row) {
        for (std::int64_t col = 0; col < kCols; ++col) {
            std::uint8_t* texel =
                &texels[static_cast<std::size_t>(row * kCols + col) *
                        4];
            const bool square = std::abs(col - kCols * 3 / 4) < 16 &&
                                std::abs(row - kRows / 2) < 16;
            texel[0] = !square && col < kCols / 2 ? 220 : 20;
            texel[1] = square ? 220 : 20;
            texel[2] = !square && col >= kCols / 2 ? 220 : 20;
            texel[3] = 255;
        }
    }
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "ernie_demo_andromeda_stand_in.png";
    app::WritePng(
        path, Tensor::Copy<std::uint8_t>(texels, {kRows, kCols, 4}));
    return path;
}

class AndromedaDemo : public app::Demo {
  public:
    AndromedaDemo(app::Viewer* viewer,
                  const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 3;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // The plane facing +z, texture coordinates (0, 0) to (1, 1)
        // from its lower left corner.
        const double x = 0.5 * kWidth;
        const double y = 0.5 * kHeight;
        const std::vector<double> points = {-x, -y, 0.0, x,  -y, 0.0,
                                            x,  y,  0.0, -x, y,  0.0};
        const std::vector<float> st = {0, 0, 1, 0, 1, 1, 0, 1};
        Mesh::Define(scene, "/World/Plane")
            .SetPoints(Tensor::Copy<double>(points, {4, 3}))
            .SetFaceVertexCounts(Tensor::Copy<std::int64_t>(
                std::vector<std::int64_t>{4}, {1}))
            .SetFaceVertexIndices(Tensor::Copy<std::int64_t>(
                std::vector<std::int64_t>{0, 1, 2, 3}, {4}))
            .SetMaterialBinding("/World/Looks/Print");
        scene.Write("/World/Plane", "primvars:st",
                    Tensor::Copy<float>(st, {4, 2}));
        Material::Define(scene, "/World/Looks/Print");
        Shader::Define(scene, "/World/Looks/Print/Surface")
            .SetInfoId(tokens::kUsdPreviewSurface)
            .SetDiffuseColor(glm::dvec3(0.05))
            .SetRoughness(1.0);
        Connect(scene, "/World/Looks/Print", tokens::kSurfaceOutput,
                "/World/Looks/Print/Surface", tokens::kSurfaceOutput);
        Shader::Define(scene, "/World/Looks/Print/St")
            .SetInfoId(tokens::kUsdPrimvarReaderFloat2)
            .SetVarname("st");
        // Lit straight on.
        DistantLight::Define(scene, "/World/Light")
            .SetIrradiance(1361.0)
            .SetShadowEnable(false)
            .SetTransform(DistantLightTransform({0.0, 0.0, 1.0}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetClippingRange(glm::dvec2(
                1e-4, std::numeric_limits<double>::infinity()))
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(16.0)
            .SetExposureTime(1.0 / 250.0);
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({1280, 720})
            .SetAmbient(0.0);

        if (options.test) {
            UseImage(StandIn());
        } else {
            SelectPicture(0);
        }
        std::vector<std::string> labels;
        for (const Picture& picture : kPictures) {
            labels.emplace_back(picture.label);
        }
        Widgets().Combo("Image", labels, 0,
                        [this](std::size_t i) { SelectPicture(i); });
        // Zoom in powers of two; pans as fractions of the image.
        Widgets().Slider("Zoom (stops)", 0.0, 10.0, zoom_,
                         [this](double stops) {
                             zoom_ = stops;
                             Place();
                         });
        Widgets().Slider("Across", 0.0, 1.0, across_,
                         [this](double t) {
                             across_ = t;
                             Place();
                         });
        Widgets().Slider("Up", 0.0, 1.0, up_, [this](double t) {
            up_ = t;
            Place();
        });
        Place();
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            app::ViewportDisplay display =
                viewer->GetViewportDisplay();
            display.lights = false;
            viewer->SetViewportDisplay(display);
        }
    }

    void Step() override {
        if (download_ == nullptr) {
            return;
        }
        for (const auto& message : download_->TakeMessages()) {
            Report(message.text, message.error);
        }
        Widgets().Set(progress_, download_->Fraction());
        Widgets().Set(progress_, download_->Status());
        if (download_->Done()) {
            if (download_->Succeeded()) {
                UseImage(downloading_);
            }
            download_.reset();
            Widgets().Remove(progress_);
        }
    }

    void TestFinal() override {
        // The whole stand-in: red on the left, blue on the right.
        zoom_ = 0.0;
        across_ = 0.5;
        up_ = 0.5;
        Place();
        const auto at = [this](double u) {
            const Tensor image = GetRenderer().Render(GetScene());
            const std::int64_t rows = image.Shape()[0];
            const std::int64_t cols = image.Shape()[1];
            const auto col = static_cast<std::int64_t>(
                u * static_cast<double>(cols));
            const auto* p =
                static_cast<const std::uint8_t*>(image.Data()) +
                ((rows / 2) * cols + col) * 4;
            return glm::dvec3(p[0], p[1], p[2]);
        };
        const glm::dvec3 left = at(0.25);
        const glm::dvec3 right = at(0.75);
        // Zoomed 8 stops onto the green square, 32 of 9000 pixels:
        // only its fine tiles show it.
        zoom_ = 8.0;
        across_ = 0.75;
        Place();
        const glm::dvec3 square = at(0.5);
        std::printf(
            "Left %.0f %.0f %.0f; right %.0f %.0f %.0f; zoomed "
            "%.0f %.0f %.0f\n",
            left.r, left.g, left.b, right.r, right.g, right.b,
            square.r, square.g, square.b);
        Check(left.r > 2.0 * left.b, "the left half is red");
        Check(right.b > 2.0 * right.r, "the right half is blue");
        Check(square.g > 2.0 * std::max(square.r, square.b),
              "zoomed in, the green square fills the view");
    }

  private:
    /// Shows picture `index`, fetching it first if need be.
    void SelectPicture(std::size_t index) {
        const Picture& picture = kPictures[index];
        const std::filesystem::path path =
            demos::CacheDir() / "andromeda" / picture.file;
        if (download_ != nullptr) {
            download_.reset();
            Widgets().Remove(progress_);
        }
        if (std::filesystem::exists(path)) {
            UseImage(path);
        } else if (demos::CanDownload()) {
            downloading_ = path;
            download_ = std::make_unique<demos::AsyncDownload>(
                picture.url, path);
            progress_ = Widgets().Progress("Download");
        } else {
            Report("The panorama is not cached and cannot be fetched",
                   true);
        }
    }

    /// Prints the plane with `path`, read as sRGB colour.
    void UseImage(const std::filesystem::path& path) {
        Scene& scene = GetScene();
        Shader::Define(scene, "/World/Looks/Print/Image")
            .SetInfoId(tokens::kUsdUVTexture)
            .SetSourceColorSpace("sRGB")
            .SetWrapS("clamp")
            .SetWrapT("clamp")
            .SetFile(path.string());
        Connect(scene, "/World/Looks/Print/Image", tokens::kSt,
                "/World/Looks/Print/St", "outputs:result");
        Connect(scene, "/World/Looks/Print/Surface",
                tokens::kDiffuseColor, "/World/Looks/Print/Image",
                "outputs:rgb");
    }

    /// The camera `zoom_` stops in from the whole plane, over the
    /// point `across_` and `up_` of the way along and up it.
    void Place() {
        const glm::dvec3 target((across_ - 0.5) * kWidth,
                                (up_ - 0.5) * kHeight, 0.0);
        const glm::dvec3 eye =
            target + glm::dvec3(0.0, 0.0, kWhole / std::exp2(zoom_));
        Camera(GetScene(), "/World/Camera")
            .SetTransform(
                LookAtTransform(eye, target, {0.0, 1.0, 0.0}));
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

    std::unique_ptr<demos::AsyncDownload> download_;
    std::filesystem::path downloading_;
    app::WidgetPanel::Id progress_ = 0;
    double zoom_ = 0.0;  ///< Stops.
    double across_ = 0.5;
    double up_ = 0.5;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    AndromedaDemo, "textures", "andromeda",
    "Hubble's half-gigapixel panorama of the Andromeda Galaxy on a "
    "plane, streamed as a virtual texture as the camera zooms in.");

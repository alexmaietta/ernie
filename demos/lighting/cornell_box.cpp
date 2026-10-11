// SPDX-License-Identifier: BSD-3-Clause
// The Cornell box from its measured geometry, wall reflectance
// spectra and light spectrum: direct light only, as yet.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

// The Cornell Program of Computer Graphics' measurements, in its
// coordinates (millimetres, y up) scaled to metres.
//
// References:
// - Cornell University Program of Computer Graphics, "Cornell Box
//   Data", 2005.
//   http://www.graphics.cornell.edu/online/box/data.html
// - C. M. Goral, K. E. Torrance, D. P. Greenberg and B. Battaile,
//   "Modeling the Interaction of Light Between Diffuse Surfaces",
//   SIGGRAPH 1984. https://doi.org/10.1145/964965.808601
constexpr double kMm = 0.001;

// Reflectance at each wavelength (nm): white, green and red.
constexpr std::array<std::array<float, 4>, 76> kReflectance = {{
    {400, 0.343f, 0.092f, 0.040f}, {404, 0.445f, 0.096f, 0.046f},
    {408, 0.551f, 0.098f, 0.048f}, {412, 0.624f, 0.097f, 0.053f},
    {416, 0.665f, 0.098f, 0.049f}, {420, 0.687f, 0.095f, 0.050f},
    {424, 0.708f, 0.095f, 0.053f}, {428, 0.723f, 0.097f, 0.055f},
    {432, 0.715f, 0.095f, 0.057f}, {436, 0.710f, 0.094f, 0.056f},
    {440, 0.745f, 0.097f, 0.059f}, {444, 0.758f, 0.098f, 0.057f},
    {448, 0.739f, 0.096f, 0.061f}, {452, 0.767f, 0.101f, 0.061f},
    {456, 0.777f, 0.103f, 0.060f}, {460, 0.765f, 0.104f, 0.062f},
    {464, 0.751f, 0.107f, 0.062f}, {468, 0.745f, 0.109f, 0.062f},
    {472, 0.748f, 0.112f, 0.061f}, {476, 0.729f, 0.115f, 0.062f},
    {480, 0.745f, 0.125f, 0.060f}, {484, 0.757f, 0.140f, 0.059f},
    {488, 0.753f, 0.160f, 0.057f}, {492, 0.750f, 0.187f, 0.058f},
    {496, 0.746f, 0.229f, 0.058f}, {500, 0.747f, 0.285f, 0.058f},
    {504, 0.735f, 0.343f, 0.056f}, {508, 0.732f, 0.390f, 0.055f},
    {512, 0.739f, 0.435f, 0.056f}, {516, 0.734f, 0.464f, 0.059f},
    {520, 0.725f, 0.472f, 0.057f}, {524, 0.721f, 0.476f, 0.055f},
    {528, 0.733f, 0.481f, 0.059f}, {532, 0.725f, 0.462f, 0.059f},
    {536, 0.732f, 0.447f, 0.058f}, {540, 0.743f, 0.441f, 0.059f},
    {544, 0.744f, 0.426f, 0.061f}, {548, 0.748f, 0.406f, 0.061f},
    {552, 0.728f, 0.373f, 0.063f}, {556, 0.716f, 0.347f, 0.063f},
    {560, 0.733f, 0.337f, 0.067f}, {564, 0.726f, 0.314f, 0.068f},
    {568, 0.713f, 0.285f, 0.072f}, {572, 0.740f, 0.277f, 0.080f},
    {576, 0.754f, 0.266f, 0.090f}, {580, 0.764f, 0.250f, 0.099f},
    {584, 0.752f, 0.230f, 0.124f}, {588, 0.736f, 0.207f, 0.154f},
    {592, 0.734f, 0.186f, 0.192f}, {596, 0.741f, 0.171f, 0.255f},
    {600, 0.740f, 0.160f, 0.287f}, {604, 0.732f, 0.148f, 0.349f},
    {608, 0.745f, 0.141f, 0.402f}, {612, 0.755f, 0.136f, 0.443f},
    {616, 0.751f, 0.130f, 0.487f}, {620, 0.744f, 0.126f, 0.513f},
    {624, 0.731f, 0.123f, 0.558f}, {628, 0.733f, 0.121f, 0.584f},
    {632, 0.744f, 0.122f, 0.620f}, {636, 0.731f, 0.119f, 0.606f},
    {640, 0.712f, 0.114f, 0.609f}, {644, 0.708f, 0.115f, 0.651f},
    {648, 0.729f, 0.117f, 0.612f}, {652, 0.730f, 0.117f, 0.610f},
    {656, 0.727f, 0.118f, 0.650f}, {660, 0.707f, 0.120f, 0.638f},
    {664, 0.703f, 0.122f, 0.627f}, {668, 0.729f, 0.128f, 0.620f},
    {672, 0.750f, 0.132f, 0.630f}, {676, 0.760f, 0.139f, 0.628f},
    {680, 0.751f, 0.144f, 0.642f}, {684, 0.739f, 0.146f, 0.639f},
    {688, 0.724f, 0.150f, 0.657f}, {692, 0.730f, 0.152f, 0.639f},
    {696, 0.740f, 0.157f, 0.635f}, {700, 0.737f, 0.159f, 0.642f},
}};

// The light's emission spectrum at each wavelength (nm), relative.
const std::vector<float> kEmission{400, 0.0f,  500, 8.0f,
                                   600, 15.6f, 700, 18.4f};

using Quad = std::array<std::array<double, 3>, 4>;

struct Surface {
    const char* name;
    int column;  // of kReflectance: 1 white, 2 green, 3 red
    std::vector<Quad> quads;
};

// Each surface's quadrilaterals, in millimetres.
const std::vector<Surface> kSurfaces = {
    {"Floor",
     1,
     {{{{552.8, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 559.2},
        {549.6, 0.0, 559.2}}}}},
    {"Ceiling",
     1,
     {{{{556.0, 548.8, 0.0},
        {556.0, 548.8, 559.2},
        {0.0, 548.8, 559.2},
        {0.0, 548.8, 0.0}}}}},
    {"BackWall",
     1,
     {{{{549.6, 0.0, 559.2},
        {0.0, 0.0, 559.2},
        {0.0, 548.8, 559.2},
        {556.0, 548.8, 559.2}}}}},
    {"RightWall",
     2,
     {{{{0.0, 0.0, 559.2},
        {0.0, 0.0, 0.0},
        {0.0, 548.8, 0.0},
        {0.0, 548.8, 559.2}}}}},
    {"LeftWall",
     3,
     {{{{552.8, 0.0, 0.0},
        {549.6, 0.0, 559.2},
        {556.0, 548.8, 559.2},
        {556.0, 548.8, 0.0}}}}},
    {"ShortBlock",
     1,
     {{{{130.0, 165.0, 65.0},
        {82.0, 165.0, 225.0},
        {240.0, 165.0, 272.0},
        {290.0, 165.0, 114.0}}},
      {{{290.0, 0.0, 114.0},
        {290.0, 165.0, 114.0},
        {240.0, 165.0, 272.0},
        {240.0, 0.0, 272.0}}},
      {{{130.0, 0.0, 65.0},
        {130.0, 165.0, 65.0},
        {290.0, 165.0, 114.0},
        {290.0, 0.0, 114.0}}},
      {{{82.0, 0.0, 225.0},
        {82.0, 165.0, 225.0},
        {130.0, 165.0, 65.0},
        {130.0, 0.0, 65.0}}},
      {{{240.0, 0.0, 272.0},
        {240.0, 165.0, 272.0},
        {82.0, 165.0, 225.0},
        {82.0, 0.0, 225.0}}}}},
    {"TallBlock",
     1,
     {{{{423.0, 330.0, 247.0},
        {265.0, 330.0, 296.0},
        {314.0, 330.0, 456.0},
        {472.0, 330.0, 406.0}}},
      {{{423.0, 0.0, 247.0},
        {423.0, 330.0, 247.0},
        {472.0, 330.0, 406.0},
        {472.0, 0.0, 406.0}}},
      {{{472.0, 0.0, 406.0},
        {472.0, 330.0, 406.0},
        {314.0, 330.0, 456.0},
        {314.0, 0.0, 456.0}}},
      {{{314.0, 0.0, 456.0},
        {314.0, 330.0, 456.0},
        {265.0, 330.0, 296.0},
        {265.0, 0.0, 296.0}}},
      {{{265.0, 0.0, 296.0},
        {265.0, 330.0, 296.0},
        {423.0, 330.0, 247.0},
        {423.0, 0.0, 247.0}}}}},
};

// The light: a 130 x 105 mm opening in the ceiling, centred over
// (278, 279.5), a hair below it so the two do not overlap.
constexpr glm::dvec3 kLightCentre(278.0, 548.7, 279.5);
constexpr glm::dvec2 kLightSize(130.0, 105.0);
constexpr double kLumens = 300.0;

/// (nm, value) pairs of kReflectance's column `column`, flat.
std::vector<float> Pairs(int column) {
    std::vector<float> pairs;
    for (const auto& row : kReflectance) {
        pairs.push_back(row[0]);
        pairs.push_back(row[column]);
    }
    return pairs;
}

/// A mesh of `quads` (millimetres) at `path`, bound to `material`.
void DefineQuads(Scene& scene, const Path& path,
                 const std::vector<Quad>& quads,
                 const Path& material) {
    std::vector<double> points;
    std::vector<std::int64_t> indices;
    for (const Quad& quad : quads) {
        for (const auto& corner : quad) {
            indices.push_back(
                static_cast<std::int64_t>(indices.size()));
            for (const double c : corner) {
                points.push_back(c * kMm);
            }
        }
    }
    const std::vector<std::int64_t> counts(quads.size(), 4);
    const auto vertices = static_cast<std::int64_t>(indices.size());
    Mesh::Define(scene, path)
        .SetPoints(Tensor::Copy<double>(points, {vertices, 3}))
        .SetFaceVertexCounts(Tensor::Copy<std::int64_t>(
            counts, {static_cast<std::int64_t>(counts.size())}))
        .SetFaceVertexIndices(
            Tensor::Copy<std::int64_t>(indices, {vertices}))
        .SetMaterialBinding(material);
}

/// The mean sRGB-encoded RGB of `image` (uint8 RGBA) over rows
/// [`row0`, `row1`) and columns [`col0`, `col1`).
glm::dvec3 MeanRgb(const Tensor& image, std::size_t row0,
                   std::size_t row1, std::size_t col0,
                   std::size_t col1) {
    const std::size_t cols = image.Shape()[1];
    const auto* data = static_cast<const std::uint8_t*>(image.Data());
    glm::dvec3 sum(0.0);
    for (std::size_t r = row0; r < row1; ++r) {
        for (std::size_t c = col0; c < col1; ++c) {
            const std::uint8_t* p = data + (r * cols + c) * 4;
            sum += glm::dvec3(p[0], p[1], p[2]);
        }
    }
    return sum / static_cast<double>((row1 - row0) * (col1 - col0));
}

class CornellBoxDemo : public app::Demo {
  public:
    CornellBoxDemo(app::Viewer* viewer,
                   const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Scope::Define(scene, "/Looks");
        const std::array<Path, 4> looks{Path(), Path("/Looks/White"),
                                        Path("/Looks/Green"),
                                        Path("/Looks/Red")};
        for (int column = 1; column <= 3; ++column) {
            Material::Define(scene, looks[column])
                .SetReflectance(Pairs(column));
        }
        for (const Surface& surface : kSurfaces) {
            DefineQuads(scene, Path("/World").Child(surface.name),
                        surface.quads, looks[surface.column]);
        }
        // The opening as the camera sees it: a panel just above the
        // light, glowing at the luminance of a Lambertian emitter of
        // its size and flux, L = flux / (pi area), in cd/m^2.
        const double luminance = kLumens / (M_PI * kLightSize.x *
                                            kLightSize.y * kMm * kMm);
        Material::Define(scene, "/Looks/Glow")
            .SetEmissiveColor(glm::dvec3(luminance));
        const double x0 = 213.0;
        const double x1 = 343.0;
        const double z0 = 227.0;
        const double z1 = 332.0;
        const double y = 548.75;
        DefineQuads(
            scene, "/World/Panel",
            {{{{x1, y, z0}, {x1, y, z1}, {x0, y, z1}, {x0, y, z0}}}},
            "/Looks/Glow");
        // Facing down, its width along x and height along z.
        const glm::dvec3 centre = kLightCentre * kMm;
        RectLight::Define(scene, "/World/Light")
            .SetWidth(kLightSize.x * kMm)
            .SetHeight(kLightSize.y * kMm)
            .SetPhysicalIlluminant(std::string(tokens::kCustom))
            .SetPhysicalCustomIlluminant(kEmission)
            .SetPhotometricPower(kLumens)
            .SetShadowEnable(true)
            .SetTransform(LookAtTransform(
                centre, centre - glm::dvec3(0.0, 1.0, 0.0),
                {0.0, 0.0, 1.0}));
        // The published camera: at (278, 273, -800) looking along +z,
        // 35 mm on a 25 mm square film, exposed automatically. The
        // box's front is 0.8 m away.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(35.0)
            .SetHorizontalAperture(25.0)
            .SetVerticalAperture(25.0)
            .SetClippingRange({0.01, 10.0})
            .SetSensorNoise(false)
            .SetExposureMode(std::string(tokens::kAuto))
            .SetTransform(LookAtTransform({0.278, 0.273, -0.8},
                                          {0.278, 0.273, 0.0},
                                          {0.0, 1.0, 0.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({512, 512})
            .SetAmbient(0.0);
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    void TestFinal() override {
        const Tensor image = RenderImage();
        const std::size_t rows = image.Shape()[0];
        const std::size_t cols = image.Shape()[1];
        // The red wall fills the image's left edge, the green its
        // right.
        const glm::dvec3 left = MeanRgb(image, rows / 3, 2 * rows / 3,
                                        cols / 32, cols / 10);
        const glm::dvec3 right =
            MeanRgb(image, rows / 3, 2 * rows / 3, cols - cols / 10,
                    cols - cols / 32);
        Check(left.r > 1.5 * std::max(left.g, left.b),
              "the left wall should be red");
        Check(right.g > 1.2 * std::max(right.r, right.b),
              "the right wall should be green");
        Check(demos::ColourDeviation(image) > 10.0,
              "the box should not render flat");
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(CornellBoxDemo, "lighting", "cornell box",
                    "The Cornell box from its measured data.");

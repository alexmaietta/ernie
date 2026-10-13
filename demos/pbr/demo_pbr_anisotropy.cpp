// SPDX-License-Identifier: BSD-3-Clause
// OpenPBR's specular anisotropy: brushed aluminium. Its lobe is
// rougher along the surface's tangent, the way its texture
// coordinates' u grows (specular_roughness_anisotropy), so highlights
// stretch along u, across the grooves. Each mesh carries texture
// coordinates: a plate brushed side to side (u up it), a disc brushed
// in circles (u outwards, so its highlights streak along radii) and a
// ball whose own u runs round it. A lamp in front of each lights a
// highlight on it.
//
// References:
// - J. Andersson, A. Portsmouth et al., "OpenPBR Surface
//   Specification", v1.1, 2024 (specular_roughness_anisotropy).
// - B. Burley, "Physically-Based Shading at Disney", SIGGRAPH 2012
//   course (anisotropic GGX).

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
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kWidth = 800;
constexpr int kHeight = 360;
constexpr double kFocal = 50.0;     // mm
constexpr double kAperture = 36.0;  // mm, horizontal
constexpr glm::dvec3 kEye(0.0, -6.5, 1.6);
constexpr glm::dvec3 kTarget(0.0, 0.0, 0.7);
constexpr double kAnisotropy = 0.7;
constexpr double kRoughness = 0.35;
constexpr double kSky = 0.4;  // the studio's intensity
constexpr double kSun = 0.6;
constexpr double kTilt = 35.0;  // degrees the panels lean back from upright
constexpr double kRadius = 0.6;  // the ball's
constexpr std::array<glm::dvec3, 3> kCentres = {
    glm::dvec3(-1.6, 0.0, 0.75), glm::dvec3(0.0, 0.0, 0.75),
    glm::dvec3(1.6, 0.0, 0.65)};
constexpr glm::dvec3 kAluminium(0.92, 0.92, 0.93);

/// A latitude-longitude studio without soft boxes, which the lamps
/// light instead: a sky brightening upwards over a dim floor, linear
/// RGB.
Tensor Studio(int rows = 64, int cols = 128) {
    std::vector<float> texels(static_cast<std::size_t>(rows * cols * 3));
    for (int i = 0; i < rows; ++i) {
        const double e = 90.0 - 180.0 * (i + 0.5) / rows;
        const double sky =
            e > 0.0 ? 0.2 + 0.6 * std::sin(e * std::numbers::pi / 180.0)
                    : 0.03;
        for (int j = 0; j < cols * 3; ++j) {
            texels[static_cast<std::size_t>(i * cols * 3 + j)] =
                static_cast<float>(sky);
        }
    }
    return Tensor::Copy<float>(texels, {rows, cols, 3});
}

/// A panel's right, up and normal: leaning back by kTilt, facing the
/// camera.
std::array<glm::dvec3, 3> PanelAxes() {
    const double t = kTilt * std::numbers::pi / 180.0;
    return {glm::dvec3(1.0, 0.0, 0.0),
            glm::dvec3(0.0, std::sin(t), std::cos(t)),
            glm::dvec3(0.0, -std::cos(t), std::sin(t))};
}

/// A mesh's points, face counts, indices and texture coordinates.
struct MeshArrays {
    std::vector<double> points;
    std::vector<std::int64_t> counts;
    std::vector<std::int64_t> indices;
    std::vector<float> st;
};

/// Appends the point (x, y) in a panel's plane at `centre`.
void AddPanelPoint(MeshArrays& mesh, const glm::dvec3& centre, double x,
                   double y) {
    const auto [right, up, normal] = PanelAxes();
    const glm::dvec3 p = centre + x * right + y * up;
    mesh.points.insert(mesh.points.end(), {p.x, p.y, p.z});
}

/// A square plate, u up it.
MeshArrays Plate(const glm::dvec3& centre, double size = 1.0) {
    MeshArrays mesh;
    const std::array<glm::dvec2, 4> corners = {
        glm::dvec2(0.0, 0.0), glm::dvec2(1.0, 0.0), glm::dvec2(1.0, 1.0),
        glm::dvec2(0.0, 1.0)};
    for (const glm::dvec2& corner : corners) {
        AddPanelPoint(mesh, centre, (corner.x - 0.5) * size,
                      (corner.y - 0.5) * size);
        mesh.st.insert(mesh.st.end(), {static_cast<float>(corner.y),
                                       static_cast<float>(corner.x)});
    }
    mesh.counts = {4};
    mesh.indices = {0, 1, 2, 3};
    return mesh;
}

/// An annulus of quads, u outwards and v round it (its seam's points
/// doubled).
MeshArrays Disc(const glm::dvec3& centre, double inner = 0.05,
                double outer = 0.6, int rings = 16, int segments = 256) {
    MeshArrays mesh;
    for (int i = 0; i <= rings; ++i) {
        const double u = static_cast<double>(i) / rings;
        const double radius = inner + (outer - inner) * u;
        for (int j = 0; j <= segments; ++j) {
            const double v = static_cast<double>(j) / segments;
            const double angle = 2.0 * std::numbers::pi * v;
            AddPanelPoint(mesh, centre, radius * std::cos(angle),
                          radius * std::sin(angle));
            mesh.st.insert(mesh.st.end(), {static_cast<float>(u),
                                           static_cast<float>(v)});
        }
    }
    // Counter-clockwise seen from the front: out, round, then in.
    for (int i = 0; i < rings; ++i) {
        for (int j = 0; j < segments; ++j) {
            const std::int64_t first = i * (segments + 1) + j;
            mesh.counts.push_back(4);
            mesh.indices.insert(mesh.indices.end(),
                                {first, first + segments + 1,
                                 first + segments + 2, first + 1});
        }
    }
    return mesh;
}

/// Where a lamp shows its reflection at `point` (of `normal`) to the
/// camera.
glm::dvec3 LampFor(const glm::dvec3& point, const glm::dvec3& normal,
                   double distance = 2.0) {
    const glm::dvec3 view = glm::normalize(kEye - point);
    const glm::dvec3 reflected =
        2.0 * glm::dot(normal, view) * normal - view;
    return point + distance * reflected;
}

/// The ball's normal where its lamp shows: on its upper front.
glm::dvec3 BallNormal() {
    return glm::normalize(glm::normalize(kEye - kCentres[2]) +
                          glm::dvec3(0.0, 0.0, 0.5));
}

/// The point on the ball where its lamp shows.
glm::dvec3 BallHighlight() {
    return kCentres[2] + kRadius * BallNormal();
}

/// A lamp for each subject: on the plate's middle, up the disc and on
/// the ball's upper front.
std::array<glm::dvec3, 3> Lamps() {
    const auto [right, up, normal] = PanelAxes();
    return {LampFor(kCentres[0], normal),
            LampFor(kCentres[1] + 0.3 * up, normal),
            LampFor(BallHighlight(), BallNormal())};
}

/// The pixel (col, row) of a world point.
glm::dvec2 Project(const glm::dvec3& point) {
    const glm::dvec3 forward = glm::normalize(kTarget - kEye);
    const glm::dvec3 right =
        glm::normalize(glm::cross(forward, glm::dvec3(0.0, 0.0, 1.0)));
    const glm::dvec3 up = glm::cross(right, forward);
    const glm::dvec3 d = point - kEye;
    const double scale =
        kFocal / kAperture * kWidth / glm::dot(d, forward);
    return {0.5 * kWidth + scale * glm::dot(d, right),
            0.5 * kHeight - scale * glm::dot(d, up)};
}

/// A highlight's spread across the image over its spread down it,
/// within `half` pixels of `point`: the second moments of the pixels
/// brighter than the window's mean, weighted by how much.
double Stretch(const Tensor& image, const glm::dvec3& point,
               int half = 40) {
    const glm::dvec2 at = Project(point);
    const int col0 = static_cast<int>(at.x) - half;
    const int row0 = static_cast<int>(at.y) - half;
    const auto* texels = static_cast<const std::uint8_t*>(image.Data());
    const std::int64_t cols = image.Shape()[1];
    const std::int64_t channels = image.Shape()[2];
    const int size = 2 * half;
    std::vector<double> brightness(static_cast<std::size_t>(size * size));
    double mean = 0.0;
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            const std::uint8_t* texel =
                texels + ((row0 + r) * cols + col0 + c) * channels;
            const double b = (texel[0] + texel[1] + texel[2]) / 3.0;
            brightness[static_cast<std::size_t>(r * size + c)] = b;
            mean += b;
        }
    }
    mean /= static_cast<double>(size * size);
    double total = 0.0;
    double meanRow = 0.0;
    double meanCol = 0.0;
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            const double w = std::max(
                brightness[static_cast<std::size_t>(r * size + c)] - mean,
                0.0);
            total += w;
            meanRow += w * r;
            meanCol += w * c;
        }
    }
    meanRow /= total;
    meanCol /= total;
    double across = 0.0;
    double down = 0.0;
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            const double w = std::max(
                brightness[static_cast<std::size_t>(r * size + c)] - mean,
                0.0);
            across += w * (c - meanCol) * (c - meanCol);
            down += w * (r - meanRow) * (r - meanRow);
        }
    }
    return std::sqrt(across / down);
}

void DefineMesh(Scene& scene, const Path& path, const MeshArrays& mesh,
                const Path& material) {
    const auto points = static_cast<std::int64_t>(mesh.points.size() / 3);
    Mesh::Define(scene, path)
        .SetPoints(Tensor::Copy<double>(mesh.points, {points, 3}))
        .SetFaceVertexCounts(Tensor::Copy<std::int64_t>(
            mesh.counts, {static_cast<std::int64_t>(mesh.counts.size())}))
        .SetFaceVertexIndices(Tensor::Copy<std::int64_t>(
            mesh.indices,
            {static_cast<std::int64_t>(mesh.indices.size())}))
        .SetMaterialBinding(material);
    scene.Write(path, "primvars:st",
                Tensor::Copy<float>(mesh.st, {points, 2}));
}

class AnisotropyDemo : public app::Demo {
  public:
    AnisotropyDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Floor")
            .SetChecker(true)
            .SetCheckerSize(0.5)
            .SetCheckerColor0(glm::dvec3(0.06))
            .SetCheckerColor1(glm::dvec3(0.03));
        DomeLight::Define(scene, "/World/Sky")
            .SetIntensity(kSky)
            .SetTexture(Studio());
        DistantLight::Define(scene, "/World/Sun")
            .SetIntensity(kSun)
            .SetAngle(1.0)
            .SetTransform(DistantLightTransform({0.5, -0.6, 0.62}));
        const std::array<glm::dvec3, 3> lamps = Lamps();
        for (std::size_t i = 0; i < lamps.size(); ++i) {
            const std::string lamp = "/World/Lamp" + std::to_string(i);
            SphereLight::Define(scene, lamp)
                .SetRadius(0.04)
                .SetIntensity(150.0);
            SetXform(scene, lamp, {.translate = lamps[i]});
        }
        const Path look("/World/Looks/Brushed");
        Material::Define(scene, look)
            .SetBrdf("open_pbr")
            .SetAlbedo(kAluminium);
        scene.Set(look, "inputs:base_metalness", 1.0);
        scene.Set(look, "inputs:specular_roughness", kRoughness);
        scene.Set(look, "inputs:specular_roughness_anisotropy",
                  kAnisotropy);
        DefineMesh(scene, "/World/Plate", Plate(kCentres[0]), look);
        DefineMesh(scene, "/World/Disc", Disc(kCentres[1]), look);
        // Finely tessellated: its tangents are each triangle's.
        Sphere::Define(scene, "/World/Ball")
            .SetRadius(kRadius)
            .SetTessellationLevel(256)
            .SetMaterialBinding(look);
        SetXform(scene, "/World/Ball", {.translate = kCentres[2]});
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(kFocal)
            .SetHorizontalAperture(kAperture)
            .SetVerticalAperture(kAperture * kHeight / kWidth)
            .SetSensorNoise(false)
            .SetTransform(
                LookAtTransform(kEye, kTarget, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({kWidth, kHeight})
            .SetAmbient(0.0);

        Widgets().Slider("Anisotropy", 0.0, 1.0, kAnisotropy,
                         [this](double value) {
                             GetScene().Set(
                                 "/World/Looks/Brushed",
                                 "inputs:specular_roughness_anisotropy",
                                 value);
                         });
        Widgets().Slider("Roughness", 0.0, 1.0, kRoughness,
                         [this](double value) {
                             GetScene().Set("/World/Looks/Brushed",
                                            "inputs:specular_roughness",
                                            value);
                         });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->LookAt(kEye, kTarget);
        }
    }

    void TestFinal() override {
        Scene& scene = GetScene();
        // The lamps alone.
        DomeLight(scene, "/World/Sky").SetIntensity(0.0);
        DistantLight(scene, "/World/Sun").SetIntensity(0.0);
        const glm::dvec2 brushed = Stretches();
        scene.Set("/World/Looks/Brushed",
                  "inputs:specular_roughness_anisotropy", 0.0);
        const glm::dvec2 plain = Stretches();
        scene.Set("/World/Looks/Brushed",
                  "inputs:specular_roughness_anisotropy", kAnisotropy);
        DomeLight(scene, "/World/Sky").SetIntensity(kSky);
        DistantLight(scene, "/World/Sun").SetIntensity(kSun);
        std::printf(
            "Highlights' spread across over down: plate %.2f (isotropic "
            "%.2f), ball %.2f (isotropic %.2f)\n",
            brushed.x, plain.x, brushed.y, plain.y);
        Check(brushed.x < 0.6 * plain.x,
              "the plate's highlight stretches up it, along u");
        Check(brushed.y > 1.5 * plain.y,
              "the ball's highlight stretches round it, along u");
    }

  private:
    /// The plate's and the ball's lamps' highlights' spreads across
    /// over down.
    glm::dvec2 Stretches() {
        const Tensor image = GetRenderer().Render(GetScene());
        return {Stretch(image, kCentres[0]),
                Stretch(image, BallHighlight(), 16)};
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    AnisotropyDemo, "pbr", "anisotropy",
    "OpenPBR's specular anisotropy: brushed aluminium plate, disc and "
    "ball, highlights stretched along their texture's u.");

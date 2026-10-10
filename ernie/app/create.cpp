// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/create.h"

#include "ernie/app/session.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/astronomy.h"
#include "ernie/util/tensor.h"

#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie::app {

namespace {

using namespace tokens;

constexpr CreatableType kTypes[] = {
    {kXform, "Xform", "Organize", "Xform", "/World"},
    {kScope, "Scope", "Organize", "Scope", "/World"},
    {kCube, "Cube", "Geometry", "Cube", "/World"},
    {kSphere, "Sphere", "Geometry", "Sphere", "/World"},
    {kCylinder, "Cylinder", "Geometry", "Cylinder", "/World"},
    {kCone, "Cone", "Geometry", "Cone", "/World"},
    {kCapsule, "Capsule", "Geometry", "Capsule", "/World"},
    {kPlane, "Plane", "Geometry", "Plane", "/World"},
    {kMesh, "Mesh (Quad)", "Geometry", "Mesh", "/World"},
    {kPointInstancer, "Point Instancer", "Geometry", "Instancer", "/World"},
    {kGroundPlane, "Ground Plane", "Geometry", "Ground", "/World"},
    {kTerrain, "Terrain", "Geometry", "Terrain", "/World"},
    {kDistantLight, "Distant Light", "Lights", "Sun", "/World"},
    {kSphereLight, "Sphere Light", "Lights", "SphereLight", "/World"},
    {kRectLight, "Rect Light", "Lights", "RectLight", "/World"},
    {kDiskLight, "Disk Light", "Lights", "DiskLight", "/World"},
    {kDomeLight, "Dome Light", "Lights", "Dome", "/World"},
    {kStarField, "Star Field", "Lights", "Stars", "/World"},
    {kCamera, "Camera from View", "Cameras", "Camera", "/World"},
    {kMaterial, "Material", "Shading", "Material", "/Looks"},
    {kRenderSettings, "Render Settings", "Render", "Settings", "/Render"},
    {kRenderProduct, "Render Product", "Render", "Product", "/Render"},
    {kRenderVar, "Render Var", "Render", "Var", "/Render/Vars"},
};

/// Places the prim at `path` at world `at`, scaled by `scale`, under its parent.
void Place(Scene& scene, const Path& path, const CreateContext& context, const glm::dvec3& at,
           double scale) {
    const glm::dmat4 world = glm::scale(glm::translate(glm::dmat4(1.0), at), glm::dvec3(scale));
    scene.Set(path, kTransform, glm::inverse(context.parentWorld) * world);
}

/// Places a light `height` view scales above the target, looking down at it.
void PlaceAbove(Scene& scene, const Path& path, const CreateContext& context, double height) {
    const glm::dvec3 eye = context.target + glm::dvec3(0.0, 0.0, height * context.scale);
    const glm::dmat4 world = LookAtTransform(eye, context.target, {0.0, 1.0, 0.0});
    scene.Set(path, kTransform, glm::inverse(context.parentWorld) * world);
}

/// The units the stage's own lights (not the viewer's session lights) come in.
enum class StageLightUnits { None, Relative, Photometric };

StageLightUnits LightUnitsOf(const Scene& scene, const Path& created) {
    StageLightUnits units = StageLightUnits::None;
    for (const std::string_view type :
         {kDistantLight, kSphereLight, kRectLight, kDiskLight, kDomeLight}) {
        for (const Path& light : scene.Prims(type)) {
            if (light == created || IsSessionPath(light)) {
                continue;
            }
            if (scene.Get(light, kPhotometricIlluminance) != nullptr ||
                scene.Get(light, kPhotometricIlluminanceDistance) != nullptr ||
                scene.Get(light, kPhotometricPower) != nullptr) {
                return StageLightUnits::Photometric;
            }
            units = StageLightUnits::Relative;
        }
    }
    return units;
}

}  // namespace

std::span<const CreatableType> CreatableTypes() {
    return kTypes;
}

const CreatableType* FindCreatable(std::string_view type) {
    for (const CreatableType& creatable : kTypes) {
        if (creatable.type == type) {
            return &creatable;
        }
    }
    return nullptr;
}

void AuthorDefaultPrim(Scene& scene, const Path& path, std::string_view type,
                       const CreateContext& context) {
    if (FindCreatable(type) == nullptr) {
        throw std::invalid_argument("The viewer cannot create a '" + std::string(type) + "'");
    }
    scene.Define(path, type);
    const double s = context.scale;
    if (type == kCube || type == kSphere || type == kCylinder || type == kCone ||
        type == kCapsule || type == kPlane) {
        Place(scene, path, context, context.target, s);
        scene.Set(path, kDisplayColor, glm::vec3(0.7f));
    } else if (type == kXform) {
        Place(scene, path, context, context.target, 1.0);
    } else if (type == kMesh) {
        const std::array<float, 12> points{-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0};
        const std::array<std::int64_t, 1> counts{4};
        const std::array<std::int64_t, 4> indices{0, 1, 2, 3};
        scene.Write(path, kPoints, Tensor::Copy<float>(points, {4, 3}));
        scene.Write(path, kFaceVertexCounts, Tensor::Copy<std::int64_t>(counts, {1}));
        scene.Write(path, kFaceVertexIndices, Tensor::Copy<std::int64_t>(indices, {4}));
        scene.Set(path, kDisplayColor, glm::vec3(0.7f));
        Place(scene, path, context, context.target, s);
    } else if (type == kPointInstancer) {
        // A 5 x 5 grid of spheres.
        std::vector<float> positions;
        for (int y = -2; y <= 2; ++y) {
            for (int x = -2; x <= 2; ++x) {
                positions.insert(positions.end(),
                                 {static_cast<float>(x), static_cast<float>(y), 0.0f});
            }
        }
        const Path prototype = path.Child("Prototype");
        scene.Define(prototype, kSphere).Set(kGeomRadius, 0.3).Set(kDisplayColor, glm::vec3(0.7f));
        scene.Write(path, kPositions, Tensor::Copy<float>(positions, {25, 3}));
        scene.Set(path, kPrototypes, std::vector<Path>{prototype});
        Place(scene, path, context, context.target, s);
    } else if (type == kGroundPlane) {
        scene.Set(path, kGroundHeight, context.target.z);
    } else if (type == kTerrain) {
        // 65 x 65 cells of gentle relief, centred on the target.
        constexpr int kCells = 65;
        std::vector<float> heights(kCells * kCells);
        for (int row = 0; row < kCells; ++row) {
            for (int col = 0; col < kCells; ++col) {
                heights[static_cast<std::size_t>(row * kCells + col)] = static_cast<float>(
                    context.target.z + s * (2.0 * std::sin(col / 9.0) * std::cos(row / 11.0)));
            }
        }
        const double cell = 0.5 * s;
        scene.Write(path, kHeights, Tensor::Copy<float>(heights, {kCells, kCells}));
        scene.Set(path, kOrigin,
                  glm::dvec2(context.target.x - 0.5 * kCells * cell,
                             context.target.y + 0.5 * kCells * cell));
        scene.Set(path, kCellSize, cell);
    } else if (type == kDistantLight) {
        // Next to photometric lights, a clear-day sun; otherwise a relative sun of 1, which
        // lights white to 1 without exposure.
        scene.Set(path, kTransform, DistantLightTransform({0.4, -0.5, 0.75}));
        if (LightUnitsOf(scene, path) == StageLightUnits::Photometric) {
            scene.Set(path, kPhotometricIlluminance, 100000.0);
        }
    } else if (type == kDomeLight) {
        // A clear sky gives about a tenth of the sun's light: 10000 lux next to photometric
        // lights, 0.1 next to relative ones, and 1 when it is the stage's only light.
        switch (LightUnitsOf(scene, path)) {
        case StageLightUnits::Photometric:
            scene.Set(path, kPhotometricIlluminance, 10000.0);
            break;
        case StageLightUnits::Relative:
            scene.Set(path, kIntensity, 0.1);
            break;
        case StageLightUnits::None:
            break;
        }
    } else if (type == kStarField) {
        // A made-up naked-eye sky until a catalogue replaces it.
        const StarCatalogue stars = SyntheticStars(5000);
        const auto write = [&](std::string_view attribute, const std::vector<float>& values) {
            scene.Write(path, attribute,
                        Tensor::Copy<float>(values, {static_cast<std::int64_t>(values.size())}));
        };
        write(kStarsRightAscension, stars.rightAscension);
        write(kStarsDeclination, stars.declination);
        write(kStarsMagnitude, stars.magnitude);
        write(kStarsColorIndex, stars.colorIndex);
    } else if (type == kSphereLight) {
        PlaceAbove(scene, path, context, 3.0);
        scene.Set(path, kRadius, 0.1 * s);
        scene.Set(path, kPhotometricPower, 1600.0 * s * s);
    } else if (type == kRectLight || type == kDiskLight) {
        PlaceAbove(scene, path, context, 3.0);
        scene.Set(path, kPhotometricPower, 1600.0 * s * s);
        if (type == kRectLight) {
            scene.Set(path, kWidth, s);
            scene.Set(path, kHeight, s);
        } else {
            scene.Set(path, kRadius, 0.5 * s);
        }
    } else if (type == kCamera) {
        CameraParams camera = context.camera;
        camera.transform = glm::inverse(context.parentWorld) * camera.transform;
        SetCamera(scene, path, camera);
    } else if (type == kMaterial) {
        scene.Set(path, kAlbedo, glm::vec3(0.5f));
    } else if (type == kRenderProduct) {
        if (!context.activeCamera.IsEmpty()) {
            scene.Set(path, kCameraRel, context.activeCamera);
        }
        scene.Set(path, kResolution, glm::ivec2(1280, 720));
    } else if (type == kRenderVar) {
        scene.Set(path, kSourceName, std::string(kLdrColor));
    } else if (type == kRenderSettings) {
        if (!context.activeCamera.IsEmpty()) {
            scene.Set(path, kCameraRel, context.activeCamera);
        }
    }
}

}  // namespace ernie::app

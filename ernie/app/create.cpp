// SPDX-License-Identifier: BSD-3-Clause
/// @file create.cpp
/// @brief The Create menu's types and the defaults each new prim is authored with.
#include "ernie/app/create.h"

#include "ernie/scene/schema/camera.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/lux.h"
#include "ernie/scene/schema/mesh.h"
#include "ernie/scene/schema/planet.h"
#include "ernie/scene/schema/render.h"
#include "ernie/scene/schema/shade.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ernie::app {
namespace {

constexpr std::array kTypes{
    CreatableType{Xform::kName, "Xform", "Organize", "Xform", "/World"},
    CreatableType{Scope::kName, "Scope", "Organize", "Scope", "/World"},
    CreatableType{Cube::kName, "Cube", "Geometry", "Cube", "/World"},
    CreatableType{Sphere::kName, "Sphere", "Geometry", "Sphere", "/World"},
    CreatableType{Cylinder::kName, "Cylinder", "Geometry", "Cylinder", "/World"},
    CreatableType{Cone::kName, "Cone", "Geometry", "Cone", "/World"},
    CreatableType{Capsule::kName, "Capsule", "Geometry", "Capsule", "/World"},
    CreatableType{Plane::kName, "Plane", "Geometry", "Plane", "/World"},
    CreatableType{Mesh::kName, "Mesh (Quad)", "Geometry", "Mesh", "/World"},
    CreatableType{GroundPlane::kName, "Ground Plane", "Geometry", "Ground", "/World"},
    CreatableType{DistantLight::kName, "Distant Light", "Lights", "Sun", "/World"},
    CreatableType{SphereLight::kName, "Sphere Light", "Lights", "SphereLight", "/World"},
    CreatableType{RectLight::kName, "Rect Light", "Lights", "RectLight", "/World"},
    CreatableType{DiskLight::kName, "Disk Light", "Lights", "DiskLight", "/World"},
    CreatableType{CylinderLight::kName, "Cylinder Light", "Lights", "CylinderLight", "/World"},
    CreatableType{DomeLight::kName, "Dome Light", "Lights", "Dome", "/World"},
    CreatableType{Camera::kName, "Camera from View", "Cameras", "Camera", "/World"},
    CreatableType{Material::kName, "Material", "Shading", "Material", "/Looks"},
    CreatableType{RenderSettings::kName, "Render Settings", "Render", "Settings", "/Render"},
    CreatableType{RenderProduct::kName, "Render Product", "Render", "Product", "/Render"},
    CreatableType{RenderVar::kName, "Render Var", "Render", "Var", "/Render/Vars"},
};

/// @brief Places the prim at world `at`, scaled by `scale`, under its parent.
void Place(Scene& scene, const Path& path, const CreateContext& context, const glm::dvec3& at,
           double scale) {
    const glm::dmat4 world = glm::scale(glm::translate(glm::dmat4(1.0), at), glm::dvec3(scale));
    scene.Set(path, tokens::kTransform, glm::inverse(context.parentWorld) * world);
}

/// @brief Places a light three view scales above the target, facing down at it.
void PlaceAbove(Scene& scene, const Path& path, const CreateContext& context) {
    const glm::dvec3 eye = context.target + glm::dvec3(0.0, 0.0, 3.0 * context.scale);
    scene.Set(path, tokens::kTransform,
              glm::inverse(context.parentWorld) *
                  LookAtTransform(eye, context.target, {0.0, 1.0, 0.0}));
}

void AuthorQuad(Scene& scene, const Path& path) {
    const std::array<float, 12> points{-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0};
    const std::array<std::int32_t, 1> counts{4};
    const std::array<std::int32_t, 4> indices{0, 1, 2, 3};
    scene.Write(scene.Require(path), Mesh::kPoints,
                Tensor::Copy<float>(std::span<const float>(points), {4, 3}));
    scene.Write(scene.Require(path), Mesh::kFaceVertexCounts,
                Tensor::Copy<std::int32_t>(std::span<const std::int32_t>(counts), {1}));
    scene.Write(scene.Require(path), Mesh::kFaceVertexIndices,
                Tensor::Copy<std::int32_t>(std::span<const std::int32_t>(indices), {4}));
}

/// @brief Sizes an area light to the view's scale.
void SizeAreaLight(Scene& scene, const Path& path, std::string_view type, double s) {
    if (type == SphereLight::kName) {
        SphereLight(scene, path).SetRadius(0.1 * s);
    } else if (type == RectLight::kName) {
        RectLight(scene, path).SetWidth(s).SetHeight(s);
    } else if (type == DiskLight::kName) {
        DiskLight(scene, path).SetRadius(0.5 * s);
    } else if (type == CylinderLight::kName) {
        CylinderLight(scene, path).SetLength(s).SetRadius(0.05 * s);
    }
}

/// @brief Gives a camera the view's pose and field of view, on its default film.
void AuthorCameraFromView(Scene& scene, const Path& path, const CreateContext& context) {
    Camera camera(scene, path);
    const double aperture = camera.GetVerticalAperture();
    camera.SetFocalLength(0.5 * aperture / std::tan(0.5 * context.verticalFov));
    scene.Set(path, tokens::kTransform, glm::inverse(context.parentWorld) * context.view);
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
    const bool areaLight = type == SphereLight::kName || type == RectLight::kName ||
                           type == DiskLight::kName || type == CylinderLight::kName;
    if (type == Mesh::kName) {
        AuthorQuad(scene, path);
    }
    if (type == Cube::kName || type == Sphere::kName || type == Cylinder::kName ||
        type == Cone::kName || type == Capsule::kName || type == Plane::kName ||
        type == Mesh::kName) {
        Place(scene, path, context, context.target, s);
        scene.Set(path, Gprim::kDisplayColor, glm::dvec3(0.7));
    } else if (type == Xform::kName) {
        Place(scene, path, context, context.target, 1.0);
    } else if (type == GroundPlane::kName) {
        Place(scene, path, context, glm::dvec3(0.0, 0.0, context.target.z), 1.0);
    } else if (type == DistantLight::kName) {
        scene.Set(path, tokens::kTransform, DistantLightTransform({0.4, -0.5, 0.75}));
    } else if (areaLight) {
        PlaceAbove(scene, path, context);
        SizeAreaLight(scene, path, type, s);
    } else if (type == Camera::kName) {
        AuthorCameraFromView(scene, path, context);
    } else if ((type == RenderSettings::kName || type == RenderProduct::kName) &&
               !context.camera.IsEmpty()) {
        scene.Set(path, RenderSettingsBase::kCamera, context.camera);
    } else if (type == RenderVar::kName) {
        RenderVar(scene, path).SetSourceName(std::string(render_vars::kLdrColor));
    }
}

Path FreeChildPath(const Scene& scene, const Path& parent, std::string_view name) {
    Path path = parent.Child(name);
    for (int number = 1; scene.Find(path); ++number) {
        path = parent.Child(std::string(name) + std::to_string(number));
    }
    return path;
}

}  // namespace ernie::app

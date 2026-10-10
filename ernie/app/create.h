// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/camera.h"
#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"

#include <glm/glm.hpp>

#include <span>
#include <string_view>

namespace ernie::app {

/// A prim type the viewer's Create menu makes, with its menu label and group, the name new prims
/// take, and the prim they go under unless a group prim is selected.
struct CreatableType {
    std::string_view type;
    const char* label;
    const char* group;  ///< "Geometry", "Lights", "Cameras", "Shading", "Render" or "Organize".
    const char* name;
    const char* parent;
};

/// Every type the Create menu offers, by group.
std::span<const CreatableType> CreatableTypes();

/// The type the Create menu offers as `type`; null if it offers none.
const CreatableType* FindCreatable(std::string_view type);

/// Where a new prim goes and what it starts from.
struct CreateContext {
    glm::dvec3 target{0.0};       ///< The world point the view looks at: geometry goes there.
    double scale = 1.0;           ///< World units per default size, from the view's distance.
    glm::dmat4 parentWorld{1.0};  ///< The new prim's parent's world transform.
    CameraParams camera;          ///< The view, for new cameras.
    Path activeCamera;            ///< The camera new render products look through.
};

/// Defines a prim of `type` at `path` in `scene` (a scratch scene, copied into the real one as
/// one undo step) with defaults that make it show: geometry at the view's target at the view's
/// scale, lights above it shining down, a camera with the view's pose and lens, a terrain of
/// gentle relief, a point instancer of a few spheres, and render prims wired to the active camera.
/// @throws std::invalid_argument for a type `CreatableTypes` does not list.
void AuthorDefaultPrim(Scene& scene, const Path& path, std::string_view type,
                       const CreateContext& context);

}  // namespace ernie::app

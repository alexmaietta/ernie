// SPDX-License-Identifier: BSD-3-Clause
/// @file create.h
/// @brief The viewer's Create menu: the prim types it makes, and the defaults that make each show.
#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <span>
#include <string_view>

namespace ernie::app {

/// @brief A prim type the Create menu makes.
struct CreatableType {
    std::string_view type;
    const char* label;
    /// The menu's section: Organize, Geometry, Lights, Cameras, Shading or Render.
    const char* group;
    /// The name a new prim takes, numbered when taken.
    const char* name;
    /// Where it goes unless an Xform or Scope is selected.
    const char* parent;
};

/// @return Every type the Create menu offers, section by section.
std::span<const CreatableType> CreatableTypes();

/// @return The type's Create menu entry, or null.
const CreatableType* FindCreatable(std::string_view type);

/// @brief Where a new prim goes and what it starts from: the view.
struct CreateContext {
    /// The point the view looks at, where geometry goes.
    glm::dvec3 target{0.0};
    /// Metres a default size, from the view's distance, so new prims show at any scale.
    double scale = 1.0;
    /// The new prim's parent's world transform.
    glm::dmat4 parentWorld{1.0};
    /// The view's camera to world and vertical field of view, for a camera from the view.
    glm::dmat4 view{1.0};
    double verticalFov = 0.785398;
    /// The camera new render settings and products look through; empty for none.
    Path camera;
};

/// @brief Defines a prim with defaults that make it show: geometry at the view's target and
/// scale, lights above it shining down, a camera with the view's pose, render prims through the
/// view's camera.
/// @param scene A scratch scene, its prims copied in as one undo step (`EditHistory::Add`).
/// @throws std::invalid_argument for a type `CreatableTypes` does not list.
void AuthorDefaultPrim(Scene& scene, const Path& path, std::string_view type,
                       const CreateContext& context);

/// @return A child path of `parent` named `name`, numbered from 1 after it when taken.
Path FreeChildPath(const Scene& scene, const Path& parent, std::string_view name);

}  // namespace ernie::app

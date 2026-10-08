#pragma once

#include <string_view>

struct ImDrawList;
struct ImVec2;

namespace ernie::app {

/// Draws the icon of a prim of `type` into the square of `size` pixels at `corner` of `draw`:
/// small coloured vector drawings in the manner of Omniverse Kit's stage icons (an axis tripod
/// for transforms, a folder for scopes, shaded solids for geometry, a sphere for materials, yellow
/// lamps for lights, a camera, a film frame for render prims), drawn rather than taken from a font
/// so they stay crisp at any size and need no image assets.
void DrawPrimIcon(ImDrawList* draw, const ImVec2& corner, float size, std::string_view type);

/// Reserves a `size` square at the cursor, as an item, and draws `type`'s icon in it.
void PrimIcon(std::string_view type, float size);

}  // namespace ernie::app

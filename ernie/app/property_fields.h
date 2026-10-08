#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/value.h"

#include <span>
#include <string_view>
#include <vector>

namespace ernie::app {

class Viewer;

/// One known attribute a panel edits: how to show it, and what Ernie uses when it is not authored.
struct PropertyField {
    enum class Kind {
        Bool,
        Int,
        Float,
        Token,  ///< One of `options`.
        Color,  ///< Linear RGB.
        Vec2,
        Text,
    };
    std::string_view attribute;
    const char* label = "";
    Kind kind = Kind::Float;
    Value fallback;  ///< Shown, muted, while the attribute is not authored.
    double min = 0.0;  ///< Drag limits; none when equal.
    double max = 0.0;
    std::vector<std::string_view> options;
    const char* tooltip = nullptr;
};

/// Draws `fields` of the prim at `path` as rows of a two-column table, `labelWidth` pixels for
/// the labels: authored values normally and fallbacks muted, each edited through the viewer's undo
/// history (`Viewer::Edit`) and reset by a right-click on its label (`Viewer::ClearAttribute`).
/// Returns the first error an edit raised, or an empty string.
std::string DrawPropertyFields(Viewer& viewer, const Path& path,
                               std::span<const PropertyField> fields, float labelWidth);

}  // namespace ernie::app

#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/value.h"

#include <span>
#include <string>
#include <vector>

namespace ernie::app {

class Viewer;

/// One known attribute a panel edits: how to show it, and what Ernie uses when it is not authored.
struct PropertyField {
    enum class Kind {
        Bool,
        Int,
        Float,
        Token,   ///< One of `options`.
        Color,   ///< Linear RGB.
        Vec2,
        Int2,    ///< Such as a resolution.
        Text,
        Target,  ///< A relationship to one prim, of type `options[0]` when given.
    };
    std::string attribute;
    std::string label;
    Kind kind = Kind::Float;
    Value fallback;    ///< Shown, muted, while the attribute is not authored.
    double min = 0.0;  ///< Drag limits; none when equal.
    double max = 0.0;
    std::vector<std::string> options;
    std::string tooltip;
};

/// A titled group of fields, such as a prim type's "Light" or "Shape" attributes.
struct PropertySection {
    std::string title;
    std::vector<PropertyField> fields;
};

/// Draws `fields` of the prim at `path` as rows of a two-column table, `labelWidth` pixels for
/// the labels: authored values normally and fallbacks muted, each edited through the viewer's undo
/// history (`Viewer::Edit`) and reset by a right-click on its label (`Viewer::ClearAttribute`).
/// Returns the first error an edit raised, or an empty string.
std::string DrawPropertyFields(Viewer& viewer, const Path& path,
                               std::span<const PropertyField> fields, float labelWidth);

/// Draws each of `sections` under a collapsing header, open by default.
std::string DrawPropertySections(Viewer& viewer, const Path& path,
                                 std::span<const PropertySection> sections, float labelWidth);

/// The label column's width in pixels for the viewer's theme and font.
float PropertyLabelWidth(const Viewer& viewer);

}  // namespace ernie::app

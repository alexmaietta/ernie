// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/value.h"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::app {

class Viewer;

/// One known attribute a panel edits: how to show it, and what Ernie
/// uses when it is not authored.
struct PropertyField {
    enum class Kind {
        Bool,
        Int,
        Float,
        Token,   ///< One of `options`.
        Color,   ///< Linear RGB.
        Color4,  ///< Linear RGBA.
        Vec2,
        Vec3,
        Vec4,
        Int2,  ///< Such as a resolution.
        Quat,  ///< A rotation, edited as Euler angles in degrees (x,
               ///< then y, then z).
        Matrix,  ///< A 4 x 4 matrix, edited row by row (column
                 ///< vectors, translation in the last).
        Text,
        /// A file's path (an asset attribute): its text, and a button
        /// choosing it in the viewer's file dialog.
        File,
        Target,  ///< A relationship to one prim, of type `options[0]`
                 ///< when given.
        /// A float array as a curve over [0, 1], edited by dragging
        /// across it, which sets the samples under the pointer; `min`
        /// and `max` bound the values when they differ.
        Curve,
    };
    std::string attribute;
    std::string label;
    Kind kind = Kind::Float;
    Value fallback;    ///< Shown, muted, while the attribute is not
                       ///< authored.
    double min = 0.0;  ///< Drag limits; none when equal.
    double max = 0.0;
    std::vector<std::string> options;
    std::string tooltip;
};

/// A titled group of fields, such as a prim type's "Light" or "Shape"
/// attributes.
struct PropertySection {
    std::string title;
    std::vector<PropertyField> fields;
};

/// Draws `fields` of the prim at `path` as rows of a table, as
/// Omniverse's property widgets do: a label (`labelWidth` pixels;
/// muted while the attribute is unset), the value's widget, and a
/// reset square, blue once the attribute is authored, which resets it
/// to its fallback when clicked (`Viewer::ClearAttribute`; so does a
/// right-click on the label). Edits go through the viewer's undo
/// history (`Viewer::Edit`). Only fields whose label or attribute
/// contains `filter` (case aside) show. Returns the first error an
/// edit raised, or an empty string.
std::string DrawPropertyFields(Viewer& viewer, const Path& path,
                               std::span<const PropertyField> fields,
                               float labelWidth,
                               std::string_view filter = {});

/// Draws each of `sections` under a collapsing header, open by
/// default; with a `filter`, only the sections with fields that match
/// it.
std::string
DrawPropertySections(Viewer& viewer, const Path& path,
                     std::span<const PropertySection> sections,
                     float labelWidth, std::string_view filter = {});

/// Whether `label` or `attribute` contains `filter`, ignoring case;
/// always for an empty filter.
bool MatchesFilter(std::string_view label, std::string_view attribute,
                   std::string_view filter);

/// A row's reset square: filled with the accent when `authored`, an
/// outline otherwise. Returns whether it was clicked while authored.
bool ResetSquare(const Viewer& viewer, bool authored);

/// Sets up a property table `id` of a label column `labelWidth` wide,
/// a value column and a reset column; returns false (and nothing to
/// end) when it is clipped.
bool BeginPropertyTable(const char* id, float labelWidth);

/// A collapsible section bar titled `title` in the theme's header
/// colour, open by default, as Kit's collapsable frames; returns
/// whether it is open.
bool SectionHeader(const Viewer& viewer, const char* title);

/// The label column's width in pixels for the viewer's theme and
/// font.
float PropertyLabelWidth(const Viewer& viewer);

}  // namespace ernie::app

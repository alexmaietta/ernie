// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/prim_schemas.h"

#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string>

namespace ernie::app {

namespace {

using namespace tokens;
using Kind = PropertyField::Kind;

std::string S(std::string_view text) {
    return std::string(text);
}

PropertyField Field(std::string_view attribute, std::string label, Kind kind, Value fallback,
                    double min = 0.0, double max = 0.0, std::vector<std::string> options = {},
                    std::string tooltip = {}) {
    return {S(attribute), std::move(label), kind,           std::move(fallback),
            min,          max,              std::move(options), std::move(tooltip)};
}

PropertySection Common() {
    return {"Prim",
            {Field(kVisibility, "Visibility", Kind::Token, S(kInherited), 0, 0,
                   {S(kInherited), S(kInvisible)}),
             Field(kWorld, "World", Kind::Int, -1, -1, 1 << 20, {},
                   "The world the prim and its descendants are in; -1 shared by every world")}};
}

/// The field a panel edits a registered attribute with; none for arrays and tensors.
std::optional<PropertyField> FieldOf(const AttributeDef& def) {
    std::optional<Kind> kind;
    switch (def.kind) {
    case AttributeKind::Bool:
        kind = Kind::Bool;
        break;
    case AttributeKind::Int:
        kind = Kind::Int;
        break;
    case AttributeKind::Double:
        kind = Kind::Float;
        break;
    case AttributeKind::Vec2:
        kind = Kind::Vec2;
        break;
    case AttributeKind::Vec3:
        if (def.meta.color) {
            kind = Kind::Color;
        }
        break;
    case AttributeKind::Token:
        kind = def.meta.options.empty() ? Kind::Text : Kind::Token;
        break;
    case AttributeKind::Relationship:
        kind = Kind::Target;
        break;
    case AttributeKind::FloatArray:
    case AttributeKind::Tensor:
        break;
    }
    if (!kind) {
        return std::nullopt;
    }
    std::vector<std::string> options;
    for (const std::string_view option : def.meta.options) {
        options.push_back(S(option));
    }
    return Field(def.name, def.label, *kind, def.fallback, def.meta.min, def.meta.max,
                 std::move(options), S(def.meta.doc));
}

/// A registered type's sections, one per `AttributeMeta::group` in `kGroups`' order (others after,
/// as they come, then "Prim", what every prim has); within one, the type's own attributes before
/// its bases'.
std::vector<PropertySection> SectionsOf(const PrimTypeDef& type) {
    static constexpr std::array<std::string_view, 4> kGroups{"Light", "Shape", "Spectrum",
                                                             "Photometric"};
    std::vector<PropertySection> sections;
    for (const PrimTypeDef* level = &type; level != nullptr; level = level->base) {
        for (const AttributeDef& def : level->attributes) {
            std::optional<PropertyField> field = FieldOf(def);
            if (!field) {
                continue;
            }
            auto section = std::ranges::find(sections, def.meta.group, &PropertySection::title);
            if (section == sections.end()) {
                section = sections.insert(sections.end(), PropertySection{S(def.meta.group), {}});
            }
            section->fields.push_back(std::move(*field));
        }
    }
    const auto rank = [](const PropertySection& section) {
        if (section.title == "Prim") {
            return kGroups.size() + 1;
        }
        return static_cast<std::size_t>(std::ranges::find(kGroups, section.title) -
                                        kGroups.begin());
    };
    std::ranges::stable_sort(sections, {}, rank);
    return sections;
}

/// A Material's fields its declaration cannot hold: the BRDF's choices, the registry's models, and
/// a "Reflectance model" section after "Surface" with the chosen model's parameters
/// (`inputs:<param>`).
void MaterialSections(const ernie::Material& material, const BrdfRegistry& brdfs,
                      std::vector<PropertySection>& sections) {
    const auto surface = std::ranges::find(sections, S("Surface"), &PropertySection::title);
    if (surface == sections.end()) {
        return;
    }
    for (PropertyField& field : surface->fields) {
        if (field.attribute == kBrdf) {
            field.kind = Kind::Token;
            for (const BrdfModel& model : brdfs.Models()) {
                field.options.push_back(model.id);
            }
        }
    }
    PropertySection model{"Reflectance model", {}};
    if (const BrdfModel* found = brdfs.Find(material.GetBrdf())) {
        for (const BrdfParam& param : found->params) {
            std::string tooltip = param.description;
            if (!param.units.empty()) {
                tooltip += (tooltip.empty() ? "" : " ") + ("(" + param.units + ")");
            }
            model.fields.push_back(Field("inputs:" + param.name,
                                         param.label.empty() ? param.name : param.label,
                                         Kind::Float, static_cast<double>(param.defaultValue),
                                         param.minValue, param.maxValue, {}, tooltip));
        }
    }
    if (!model.fields.empty()) {
        sections.insert(surface + 1, std::move(model));
    }
}

/// A Shader's sections for its `info:id` only, such as "Texture" for a UsdUVTexture; all of them
/// for another id.
void ShaderSections(const ernie::Shader& shader, std::vector<PropertySection>& sections) {
    using Entry = std::pair<std::string_view, std::string_view>;  // (info:id, section)
    static constexpr std::array<Entry, 4> kSectionOf{{
        {kUsdPreviewSurface, "Preview surface"},
        {kUsdUVTexture, "Texture"},
        {kUsdTransform2d, "Transform 2D"},
        {kUsdPrimvarReaderFloat2, "Primvar reader"},
    }};
    const std::string id = shader.GetInfoId();
    const auto own = std::ranges::find(kSectionOf, std::string_view(id), &Entry::first);
    if (own == kSectionOf.end()) {
        return;
    }
    std::erase_if(sections, [&](const PropertySection& section) {
        return section.title != own->second &&
               std::ranges::find(kSectionOf, std::string_view(section.title), &Entry::second) !=
                   kSectionOf.end();
    });
}

/// A render settings' or product's sections with the resolution edited as whole pixels, which the
/// registry's 2-vector cannot say, and the camera and resolution leading their section.
void RenderImageFields(std::vector<PropertySection>& sections) {
    for (PropertySection& section : sections) {
        for (PropertyField& field : section.fields) {
            if (field.attribute == kResolution) {
                field.kind = Kind::Int2;
                field.fallback = Value(glm::ivec2(*field.fallback.Get<glm::dvec2>()));
            }
        }
        std::ranges::stable_partition(section.fields, [](const PropertyField& field) {
            return field.attribute == kCameraRel || field.attribute == kResolution;
        });
    }
}

/// A SensorFilter's sections with the custom filters (`filters`) among its type's choices, and
/// the parameters of its type's model as `inputs:*`: arrays as curves of their default resampled
/// to their size, numbers as floats.
void SensorFilterSections(const SensorFilterPrim& prim, const SensorFilterRegistry& filters,
                          std::vector<PropertySection>& sections) {
    const auto section = std::ranges::find(sections, "Filter", &PropertySection::title);
    if (section == sections.end()) {
        return;
    }
    const std::string current = prim.GetFilterType();
    std::vector<PropertyField> inputs;
    for (const SensorFilterModel& model : filters.Models()) {
        for (PropertyField& field : section->fields) {
            if (field.attribute == kFilterType) {
                field.options.push_back(model.id);
            }
        }
        if (model.id != current) {
            continue;
        }
        for (const SensorFilterParam& param : model.params) {
            if (param.size > 1) {
                inputs.push_back(Field("inputs:" + param.name, param.name, Kind::Curve,
                                       param.defaults.empty()
                                           ? std::vector<float>(param.size, param.defaultValue)
                                           : ResampleCurve(param.defaults, param.size),
                                       0, 0, {}, param.description));
                continue;
            }
            inputs.push_back(Field("inputs:" + param.name, param.name, Kind::Float,
                                   static_cast<double>(param.defaultValue), 0, 0, {},
                                   param.description));
        }
    }
    section->fields.insert(section->fields.end(), inputs.begin(), inputs.end());
}

/// Adds to a registered type's `sections` the fields its declaration cannot hold, which depend on
/// the prim's values: a Material's chosen BRDF's parameters, a SensorFilter's type's inputs.
void DynamicSectionsOf(const Scene& scene, const Path& path, std::string_view type,
                       const BrdfRegistry& brdfs, const SensorFilterRegistry& filters,
                       std::vector<PropertySection>& sections) {
    if (type == kMaterial) {
        MaterialSections(ernie::Material(scene, path), brdfs, sections);
    } else if (type == kShader) {
        ShaderSections(ernie::Shader(scene, path), sections);
    } else if (type == kRenderSettings || type == kRenderProduct) {
        RenderImageFields(sections);
    } else if (type == kSensorFilter) {
        SensorFilterSections(SensorFilterPrim(scene, path), filters, sections);
    }
}

}  // namespace

std::vector<PropertySection> CameraSchema() {
    std::vector<PropertySection> sections = SectionsOf(Camera::Definition());
    // The Camera panel shows the camera's own sections, not what every prim has.
    std::erase_if(sections, [](const PropertySection& section) { return section.title == "Prim"; });
    return sections;
}

std::vector<PropertySection> RenderSettingsSchema() {
    std::vector<PropertySection> sections = SectionsOf(RenderSettings::Definition());
    RenderImageFields(sections);
    // The Settings panel shows the settings' own sections, not what every prim has.
    std::erase_if(sections, [](const PropertySection& section) { return section.title == "Prim"; });
    return sections;
}

bool HasTransform(std::string_view type) {
    return type != kMaterial && type != kRenderSettings && type != kRenderProduct &&
           type != kRenderVar && type != kSensorFilter && type != kGroundPlane &&
           type != kTerrain;
}

std::vector<PropertySection> SchemaOf(const Scene& scene, const Path& path,
                                      const BrdfRegistry& brdfs,
                                      const SensorFilterRegistry& filters) {
    const std::string& type = scene.TypeOf(path);
    // Registered types' sections come from their declarations, "Prim" among them, then what
    // their declarations cannot hold (`DynamicSectionsOf`).
    if (const PrimTypeDef* registered = FindPrimType(type)) {
        std::vector<PropertySection> sections = SectionsOf(*registered);
        DynamicSectionsOf(scene, path, type, brdfs, filters, sections);
        return sections;
    }
    std::vector<PropertySection> sections;
    if (!path.IsRoot()) {
        sections.push_back(Common());
    }
    return sections;
}

}  // namespace ernie::app

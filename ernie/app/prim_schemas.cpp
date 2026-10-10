// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/prim_schemas.h"

#include "ernie/scene/camera.h"
#include "ernie/scene/material.h"
#include "ernie/scene/prim_type.h"
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

PropertySection Geometry(bool bindable = true) {
    PropertySection section{"Geometry",
                            {Field(kDisplayColor, "Display colour", Kind::Color, glm::vec3(0.5f))}};
    if (bindable) {
        section.fields.push_back(
            Field(kMaterialBinding, "Material", Kind::Target, Value(), 0, 0, {S(kMaterial)}));
    }
    return section;
}

PropertyField Axis() {
    return Field(kAxis, "Axis", Kind::Token, S("Z"), 0, 0, {"X", "Y", "Z"});
}

std::vector<PropertySection> ShapeOf(std::string_view type) {
    if (type == kCube) {
        return {{"Shape", {Field(kSize, "Size", Kind::Float, 2.0, 0.0, 1000.0)}}};
    }
    if (type == kSphere) {
        return {{"Shape", {Field(kGeomRadius, "Radius", Kind::Float, 1.0, 0.0, 1000.0)}}};
    }
    if (type == kCylinder || type == kCone) {
        return {{"Shape",
                 {Field(kGeomRadius, "Radius", Kind::Float, 1.0, 0.0, 1000.0),
                  Field(kGeomHeight, "Height", Kind::Float, 2.0, 0.0, 1000.0), Axis()}}};
    }
    if (type == kCapsule) {
        return {{"Shape",
                 {Field(kGeomRadius, "Radius", Kind::Float, 0.5, 0.0, 1000.0),
                  Field(kGeomHeight, "Height", Kind::Float, 1.0, 0.0, 1000.0), Axis()}}};
    }
    if (type == kPlane) {
        return {{"Shape",
                 {Field(kGeomWidth, "Width", Kind::Float, 2.0, 0.0, 1000.0),
                  Field(kGeomLength, "Length", Kind::Float, 2.0, 0.0, 1000.0), Axis()}}};
    }
    return {};
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
/// as they come); within one, the type's own attributes before its bases'.
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
        return std::ranges::find(kGroups, section.title) - kGroups.begin();
    };
    std::ranges::stable_sort(sections, {}, rank);
    return sections;
}

std::vector<PropertySection> MaterialOf(const Scene& scene, const Path& path,
                                        const BrdfRegistry& brdfs) {
    const MaterialSettings defaults;
    std::vector<std::string> models;
    for (const BrdfModel& model : brdfs.Models()) {
        models.push_back(model.id);
    }
    PropertySection surface{
        "Surface",
        {Field(kBrdf, "BRDF", Kind::Token, defaults.brdf, 0, 0, models),
         Field(kAlbedo, "Albedo", Kind::Color, defaults.albedo, 0, 0, {}, "Linear RGB"),
         Field(kAlbedoMapEnable, "Use albedo map", Kind::Bool, true)}};
    // The parameters of the chosen reflectance model.
    const std::string brdf = GetToken(scene, path, kBrdf, defaults.brdf);
    PropertySection model{"Reflectance model", {}};
    if (const BrdfModel* found = brdfs.Find(brdf)) {
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
    const DetailSettings detail;
    PropertySection details{
        "Detail textures",
        {Field(kDetailTileSize, "Tile size", Kind::Float, detail.tileSize, 0.0, 1000.0),
         Field(kDetailAlbedoStrength, "Albedo strength", Kind::Float, detail.albedoStrength, 0.0,
               1.0),
         Field(kDetailNormalStrength, "Normal strength", Kind::Float, detail.normalStrength, 0.0,
               4.0),
         Field(kDetailDirectXNormals, "DirectX normals", Kind::Bool, detail.directXNormals),
         Field(kDetailHexTiling, "Hex tiling", Kind::Bool, detail.hexTiling),
         Field(kDetailHexRotation, "Hex rotation", Kind::Float, detail.hexRotation, 0.0, 1.0),
         Field(kDetailHexContrast, "Hex contrast", Kind::Float, detail.hexContrast, 0.0, 1.0),
         Field(kDetailHexSharpness, "Hex sharpness", Kind::Float, detail.hexSharpness, 1.0, 32.0)}};
    std::vector<PropertySection> sections{surface};
    if (!model.fields.empty()) {
        sections.push_back(std::move(model));
    }
    sections.push_back(std::move(details));
    return sections;
}

}  // namespace

std::vector<PropertySection> CameraSchema() {
    const Camera camera;
    const SensorModel& sensor = camera.sensor;
    return {
        {"Lens",
         {Field(kProjection, "Projection", Kind::Token, S(kPerspective), 0, 0,
                {S(kPerspective), S(kOrthographic)},
                "Orthographic reads the apertures in tenths of a scene unit"),
          Field(kFocalLength, "Focal length", Kind::Float, camera.focalLengthMm, 1.0, 2000.0, {},
                "mm"),
          Field(kVerticalAperture, "Vertical aperture", Kind::Float, camera.verticalApertureMm,
                0.1, 100.0, {}, "mm; tenths of a scene unit when orthographic"),
          Field(kHorizontalApertureOffset, "Horizontal offset", Kind::Float, 0.0, -50.0, 50.0, {},
                "mm"),
          Field(kVerticalApertureOffset, "Vertical offset", Kind::Float, 0.0, -50.0, 50.0, {},
                "mm"),
          Field(kClippingRange, "Clipping range", Kind::Vec2,
                glm::dvec2(camera.nearPlane, camera.farPlane)),
          Field(kLensProjection, "Lens projection", Kind::Token, S(kPerspective), 0, 0,
                {S(kPerspective), S(kFisheye)}),
          Field(kLensVignetting, "Vignetting", Kind::Bool, camera.lens.vignetting),
          Field(kLensTransmission, "Transmission", Kind::Float, camera.lens.transmission, 0.0,
                1.0)}},
        {"Exposure",
         {Field(kCameraExposure, "Exposure", Kind::Float, camera.exposureStops, -16.0, 16.0, {},
                "Stops; scales the photographic outputs"),
          Field(kExposureTime, "Exposure time", Kind::Float, camera.exposureTime, 1e-6, 10.0, {},
                "Seconds"),
          Field(kExposureIso, "ISO", Kind::Float, camera.iso, 25.0, 102400.0),
          Field(kExposureFStop, "f-number", Kind::Float, camera.fStop, 0.5, 64.0),
          Field(kExposureResponsivity, "Responsivity", Kind::Float, camera.responsivity, 0.0,
                100.0)}},
        {"Shutter",
         {Field(kShutterOpen, "Open", Kind::Float, camera.shutterOpen, -2.0, 1.0, {},
                "Frames from the frame's time; motion blurs from here to the close"),
          Field(kShutterClose, "Close", Kind::Float, camera.shutterClose, -2.0, 1.0, {},
                "Frames from the frame's time"),
          Field(kShutterSamples, "Samples", Kind::Int, camera.shutterSamples, 1,
                Camera::kMaxShutterSamples, {}, "Sub-frames rendered across the shutter"),
          Field(kShutterRolling, "Rolling readout", Kind::Float, camera.rollingShutter, 0.0, 2.0,
                {}, "Frames between the top row's exposure and the bottom row's; 0 is global"),
          Field(kCameraScan, "Scan", Kind::Token, S(kFrameScan), 0, 0,
                {S(kFrameScan), S(kPushbroom)},
                "Pushbroom: a line scanner whose rows run from the last frame's pose to this one's"),
          Field(kPushbroomLineOffset, "Line offset", Kind::Float, camera.lineOffsetMm, -50.0, 50.0,
                {}, "mm; the pushbroom's detector line from the principal point")}},
        {"Sensor",
         {Field(kSpectralRange, "Spectral range", Kind::Vec2,
                glm::dvec2(camera.bands.minNm, camera.bands.maxNm), 0, 0, {}, "nm"),
          Field(kSpectralBands, "Bands", Kind::Int, camera.bands.count, 1,
                SpectralBands::kMaxBands),
          Field(kSensorType, "Type", Kind::Token, S("rgb"), 0, 0, {"rgb", "mono"}),
          Field(kSensorNoise, "Noise", Kind::Bool, sensor.noise),
          Field(kSensorFullWell, "Full well", Kind::Float, sensor.fullWellElectrons, 100.0, 1e6,
                {}, "Electrons"),
          Field(kSensorReadNoise, "Read noise", Kind::Float, sensor.readNoiseElectrons, 0.0, 100.0,
                {}, "Electrons"),
          Field(kSensorDarkCurrent, "Dark current", Kind::Float, sensor.darkCurrentElectrons, 0.0,
                1000.0, {}, "Electrons per second"),
          Field(kSensorPrnu, "PRNU", Kind::Float, sensor.prnu, 0.0, 0.2),
          Field(kSensorDsnu, "DSNU", Kind::Float, sensor.dsnuElectrons, 0.0, 100.0, {},
                "Electrons"),
          Field(kSensorGain, "Electrons per DN", Kind::Float, sensor.electronsPerDnAtIso100, 0.0,
                1000.0, {}, "At ISO 100; 0 for the full well over the ADC range"),
          Field(kSensorAdcBits, "ADC bits", Kind::Int, sensor.adcBits, 1, 16),
          Field(kSensorBlackLevel, "Black level", Kind::Float, sensor.blackLevelDn, 0.0, 4096.0,
                {}, "DN"),
          Field(kSensorPixelPitch, "Pixel pitch", Kind::Float, camera.orthographicPixelPitchUm,
                0.1, 100.0, {}, "Micrometres, of orthographic cameras"),
          Field(kSensorSeed, "Seed", Kind::Int, static_cast<int>(sensor.seed))}},
    };
}

std::vector<PropertySection> RenderSettingsSchema() {
    return {
        {"Image",
         {Field(kCameraRel, "Camera", Kind::Target, Value(), 0, 0, {S(kCamera)}),
          Field(kResolution, "Resolution", Kind::Int2, glm::ivec2(2048, 1080)),
          Field(kColorMode, "Terrain colour", Kind::Token, S(kShadedColormap), 0, 0,
                {S(kHillshade), S(kColormap), S(kShadedColormap)}),
          Field(kAmbient, "Ambient", Kind::Float, 0.15, 0.0, 1.0, {},
                "Fraction of sunlight on unlit sides"),
          Field(kClearColor, "Background", Kind::Color, glm::vec3(0.0f)),
          Field(kWireframe, "Wireframe", Kind::Token, S(kOff), 0, 0,
                {S(kOff), S(kOverlay), S(kHiddenLine)}, "The viewport's toolbar may override it"),
          Field(kOverlays, "Overlays", Kind::Text, S(""), 0, 0, {},
                "\"*\" or names; the viewport shows every overlay unless set"),
          Field(kFrame, "Frame", Kind::Int, 0, 0, 0, {},
                "The frame noise derives from; counts up by itself unless set")}},
        {"Culling and LOD",
         {Field(kCulling, "Culling", Kind::Token, S(kGpu), 0, 0, {S(kOff), S(kCpu), S(kGpu)}),
          Field(kOcclusion, "Occlusion", Kind::Token, S(kOff), 0, 0,
                {S(kOff), S(kOccluders), S(kTwoPass)}),
          Field(kLodMeshError, "Mesh LOD error", Kind::Float, 1.0, 0.0, 16.0, {},
                "Pixels of error a coarser mesh level may add"),
          Field(kLodEnable, "Terrain LOD", Kind::Bool, true),
          Field(kLodDetail, "Terrain detail", Kind::Float, 4.0, 1.0, 16.0),
          Field(kLodMorphStart, "Morph start", Kind::Float, 0.67, 0.0, 1.0),
          Field(kLodFreeze, "Freeze LOD", Kind::Bool, false),
          Field(kLodShowLevels, "Show LOD levels", Kind::Bool, false)}},
        {"Shadows",
         {Field(kShadowFit, "Fit", Kind::Token, S(kScene), 0, 0, {S(kScene), S(kVisible)},
                "visible: sample distribution shadow maps (the viewport's default)"),
          Field(kShadowCascades, "Cascades", Kind::Int, 3, 1, 4),
          Field(kShadowResolution, "Resolution", Kind::Int, 2048, 256, 8192),
          Field(kShadowSoftness, "Softness", Kind::Float, 1.5, 0.0, 8.0, {},
                "Filter radius in texels"),
          Field(kShadowNormalOffset, "Normal offset", Kind::Float, 1.5, 0.0, 8.0),
          Field(kShadowSplitLambda, "Split lambda", Kind::Float, 0.8, 0.0, 1.0),
          Field(kShadowShowCascades, "Show cascades", Kind::Bool, false),
          Field(kShadowWorlds, "Worlds", Kind::Token, S(kSeparate), 0, 0,
                {S(kSeparate), S(kShared), S(kMerged)}, "Batches of views of several worlds"),
          Field(kShadowWorldFit, "World fit", Kind::Token, S(kScene), 0, 0,
                {S(kScene), S(kWorldFit)}),
          Field(kShadowWorldResolution, "World resolution", Kind::Int, 0, 0, 8192, {},
                "0 for automatic"),
          Field(kShadowWorldMargin, "World margin", Kind::Float, 1.0, 0.0, 100.0)}},
    };
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
    std::vector<PropertySection> sections;
    if (type == kCube || type == kSphere || type == kCylinder || type == kCone ||
        type == kCapsule || type == kPlane) {
        sections = ShapeOf(type);
        sections.push_back(Geometry());
    } else if (type == kMesh) {
        sections.push_back(Geometry());
    } else if (type == kGroundPlane) {
        sections.push_back(
            {"Ground",
             {Field(kGroundHeight, "Height", Kind::Float, 0.0, -1e4, 1e4, {}, "World z"),
              Field(kChecker, "Checkerboard", Kind::Bool, false),
              Field(kCheckerSize, "Checker size", Kind::Float, 1.0, 0.001, 1000.0),
              Field(kCheckerColor0, "Checker colour 0", Kind::Color, glm::vec3(1.0f)),
              Field(kCheckerColor1, "Checker colour 1", Kind::Color, glm::vec3(0.5f)),
              Field(kMaterialBinding, "Material", Kind::Target, Value(), 0, 0, {S(kMaterial)})}});
    } else if (type == kTerrain) {
        sections.push_back(
            {"Terrain",
             {Field(kOrigin, "Origin", Kind::Vec2, glm::dvec2(0.0), 0, 0, {},
                    "World x and y of the north-west corner"),
              Field(kCellSize, "Cell size", Kind::Float, 1.0, 0.0, 1e5),
              Field(kExaggeration, "Exaggeration", Kind::Float, 1.0, 0.0, 100.0),
              Field(kNoData, "No data", Kind::Float, 0.0, 0, 0, {}, "Height marking missing cells"),
              Field(kMaterialBinding, "Material", Kind::Target, Value(), 0, 0, {S(kMaterial)})}});
    } else if (const PrimTypeDef* registered = FindPrimType(type)) {
        sections = SectionsOf(*registered);
    } else if (type == kStarField) {
        sections.push_back(
            {"Stars",
             {Field(kStarsMagnitudeLimit, "Magnitude limit", Kind::Float, 99.0, -30.0, 99.0, {},
                    "Stars fainter than this are left out"),
              Field(kExposure, "Exposure", Kind::Float, 0.0, -16.0, 16.0, {},
                    "Stops; scales every star by 2^exposure"),
              Field(kStarsZeroPoint, "Zero point", Kind::Float, 2.54e-6, 0.0, 1.0, {},
                    "Lux of a magnitude 0 star (Allen's Astrophysical Quantities)")}});
    } else if (type == kCamera) {
        sections = CameraSchema();
    } else if (type == kMaterial) {
        sections = MaterialOf(scene, path, brdfs);
    } else if (type == kRenderSettings) {
        sections = RenderSettingsSchema();
    } else if (type == kRenderProduct) {
        sections.push_back(
            {"Product",
             {Field(kCameraRel, "Camera", Kind::Target, Value(), 0, 0, {S(kCamera)}),
              Field(kResolution, "Resolution", Kind::Int2, glm::ivec2(2048, 1080)),
              Field(kOverlays, "Overlays", Kind::Text, S(""), 0, 0, {},
                    "\"*\" or names of overlays drawn over its LdrColor")}});
    } else if (type == kRenderVar) {
        std::vector<std::string> names;
        for (const std::string_view name : kRenderVarNames) {
            names.emplace_back(name);
        }
        sections.push_back({"Var", {Field(kSourceName, "Source", Kind::Token, S(kLdrColor), 0, 0,
                                          names)}});
    } else if (type == kSensorFilter) {
        std::vector<std::string> types;
        for (const SensorFilter filter : DefaultFilterChain()) {
            types.emplace_back(NameOf(filter));
        }
        const std::string current = GetToken(scene, path, kFilterType, "");
        PropertySection filter{"Filter", {}};
        for (const SensorFilterModel& model : filters.Models()) {
            types.push_back(model.id);
            if (model.id == current) {
                for (const SensorFilterParam& param : model.params) {
                    if (param.size > 1) {
                        // Arrays as curves, of their default resampled to their size.
                        filter.fields.push_back(Field(
                            "inputs:" + param.name, param.name, Kind::Curve,
                            param.defaults.empty()
                                ? std::vector<float>(param.size, param.defaultValue)
                                : ResampleCurve(param.defaults, param.size),
                            0, 0, {}, param.description));
                        continue;
                    }
                    filter.fields.push_back(Field("inputs:" + param.name, param.name, Kind::Float,
                                                  static_cast<double>(param.defaultValue), 0, 0,
                                                  {}, param.description));
                }
            }
        }
        filter.fields.insert(filter.fields.begin(),
                             Field(kFilterType, "Type", Kind::Token, S(""), 0, 0, types));
        sections.push_back(std::move(filter));
    }
    if (!path.IsRoot()) {
        sections.push_back(Common());
    }
    return sections;
}

}  // namespace ernie::app

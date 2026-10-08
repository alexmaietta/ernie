#include "ernie/app/prim_schemas.h"

#include "ernie/scene/camera.h"
#include "ernie/scene/material.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"

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

/// UsdLux inputs every light reads.
PropertySection LightCommon() {
    return {"Light",
            {Field(kIntensity, "Intensity", Kind::Float, 1.0, 0.0, 1e6),
             Field(kExposure, "Exposure", Kind::Float, 0.0, -16.0, 16.0, {},
                   "Stops; scales the intensity by 2^exposure"),
             Field(kColor, "Colour", Kind::Color, glm::vec3(1.0f)),
             Field(kEnableColorTemperature, "Use temperature", Kind::Bool, false),
             Field(kColorTemperature, "Temperature", Kind::Float, 6500.0, 1000.0, 20000.0, {},
                   "Kelvin"),
             Field(kShadowEnable, "Shadows", Kind::Bool, true)}};
}

std::vector<PropertySection> LightOf(std::string_view type) {
    std::vector<PropertySection> sections{LightCommon()};
    PropertySection shape{"Shape", {}};
    if (type == kDistantLight) {
        shape.fields = {Field(kAngle, "Angle", Kind::Float, 0.53, 0.0, 30.0, {},
                              "Degrees the light subtends; widens penumbrae"),
                        Field(kLightIrradiance, "Irradiance", Kind::Float, 1361.0, 0.0, 1e5, {},
                              "W/m^2 at normal incidence; the Sun's by default")};
    } else if (type == kSphereLight) {
        shape.fields = {Field(kRadius, "Radius", Kind::Float, 0.5, 0.0, 1000.0),
                        Field(kTreatAsPoint, "Point", Kind::Bool, false),
                        Field(kShapingConeAngle, "Cone angle", Kind::Float, 180.0, 0.0, 180.0, {},
                              "Degrees; 180 unshaped"),
                        Field(kShapingConeSoftness, "Cone softness", Kind::Float, 0.0, 0.0, 1.0),
                        Field(kLightPower, "Power", Kind::Float, 0.0, 0.0, 1e7, {},
                              "Radiant flux in W, in place of the intensity when set")};
    } else if (type == kRectLight) {
        shape.fields = {Field(kWidth, "Width", Kind::Float, 1.0, 0.0, 1000.0),
                        Field(kHeight, "Height", Kind::Float, 1.0, 0.0, 1000.0),
                        Field(kLightPower, "Power", Kind::Float, 0.0, 0.0, 1e7, {},
                              "Radiant flux in W, in place of the intensity when set")};
    } else if (type == kDiskLight) {
        shape.fields = {Field(kRadius, "Radius", Kind::Float, 0.5, 0.0, 1000.0),
                        Field(kLightPower, "Power", Kind::Float, 0.0, 0.0, 1e7, {},
                              "Radiant flux in W, in place of the intensity when set")};
    } else if (type == kDomeLight) {
        shape.fields = {Field(kTextureFile, "Texture file", Kind::Text, S("")),
                        Field(kVisibleInPrimaryRay, "Visible in primary ray", Kind::Bool, true,
                              0, 0, {}, "Show behind the scene where camera rays meet nothing")};
    }
    if (!shape.fields.empty()) {
        sections.push_back(std::move(shape));
    }
    sections.push_back(
        {"Spectrum",
         {Field(kPhysicalIlluminant, "Illuminant", Kind::Token, S(kWhite), 0, 0,
                {S(kWhite), S(kBlackbody), S(kIlluminantD), S(kCustom)},
                "The spectrum's shape; blackbody and illuminant D take the temperature above. "
                "Unset: a black body if the temperature is used, else flat")}});
    const std::string switches =
        "Setting any photometric value puts the light in photometric units: they alone then set "
        "its brightness";
    PropertySection photometric{"Photometric", {}};
    if (type == kDistantLight) {
        photometric.fields = {Field(kPhotometricIlluminance, "Illuminance", Kind::Float, 10000.0,
                                    0.0, 200000.0, {}, "Lux on a surface facing it. " + switches)};
    } else if (type == kDomeLight) {
        photometric.fields = {Field(kPhotometricIlluminance, "Illuminance", Kind::Float, 10000.0,
                                    0.0, 200000.0, {},
                                    "Lux on an upward-facing surface. " + switches)};
    } else {
        photometric.fields = {
            Field(kPhotometricPower, "Power", Kind::Float, 1600.0, 0.0, 1e6, {},
                  "Luminous flux in lumens. " + switches),
            Field(kPhotometricIlluminance, "Illuminance", Kind::Float, 0.0, 0.0, 200000.0, {},
                  "Lux on a surface facing it at the distance below, in place of the power"),
            Field(kPhotometricIlluminanceDistance, "At distance", Kind::Float, 0.0, 0.0, 1000.0,
                  {}, "Metres; 0 uses the power")};
    }
    sections.push_back(std::move(photometric));
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
         {Field(kFocalLength, "Focal length", Kind::Float, camera.focalLengthMm, 1.0, 2000.0, {},
                "mm"),
          Field(kVerticalAperture, "Vertical aperture", Kind::Float, camera.verticalApertureMm,
                0.1, 100.0, {}, "mm"),
          Field(kHorizontalApertureOffset, "Horizontal offset", Kind::Float, 0.0, -50.0, 50.0, {},
                "mm"),
          Field(kVerticalApertureOffset, "Vertical offset", Kind::Float, 0.0, -50.0, 50.0, {},
                "mm"),
          Field(kClippingRange, "Clipping range", Kind::Vec2,
                glm::dvec2(camera.nearPlane, camera.farPlane)),
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
    } else if (type == kDistantLight || type == kSphereLight || type == kRectLight ||
               type == kDiskLight || type == kDomeLight) {
        sections = LightOf(type);
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

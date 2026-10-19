// SPDX-License-Identifier: BSD-3-Clause
/// @file products.cpp
/// @brief Reading render products and their cameras' lenses from UsdRender and UsdGeomCamera.
#include "ernie/render/products.h"

#include "ernie/scene/schema/camera.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/render.h"
#include "ernie/util/random.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ernie {
namespace {

constexpr std::array kRenderedVars{render_vars::kLdrColor, render_vars::kDistanceToCamera,
                                   render_vars::kNormal, render_vars::kPrimId,
                                   render_vars::kAlbedo};

std::string Quote(const Path& path) {
    return "'" + path.String() + "'";
}

/// @return The film's width and height, conformed to an image `aspect` wide (pixels across times
/// their aspect ratio, over pixels down).
glm::dvec2 ConformAperture(glm::dvec2 aperture, double aspect, std::string_view policy) {
    const bool wider = aspect > aperture.x / aperture.y;
    if (policy == "adjustApertureWidth" || (policy == "expandAperture" && wider) ||
        (policy == "cropAperture" && !wider)) {
        return {aperture.y * aspect, aperture.y};
    }
    if (policy == "adjustApertureHeight" || policy == "expandAperture" ||
        policy == "cropAperture") {
        return {aperture.x, aperture.x / aspect};
    }
    // adjustPixelAspectRatio: the film stays, and pixels stretch to fill it.
    return aperture;
}

/// @return The vars a product holds: its ordered vars' source names, or LdrColor alone.
std::vector<std::string> ReadVars(const Scene& scene, const std::vector<Path>& ordered) {
    std::vector<std::string> vars;
    for (const Path& path : ordered) {
        if (!scene.Find(path) || scene.TypeName(*scene.Find(path)) != RenderVar::kName) {
            throw std::invalid_argument(Quote(path) + " is no RenderVar");
        }
        const std::string name = RenderVar(scene, path).GetSourceName();
        if (std::ranges::find(kRenderedVars, name) == kRenderedVars.end()) {
            throw std::invalid_argument(Quote(path) + "'s source '" + name +
                                        "' is no output the Minimal tier renders");
        }
        vars.push_back(name);
    }
    if (vars.empty()) {
        vars.emplace_back(render_vars::kLdrColor);
    }
    return vars;
}

/// @return The product a settings or product prim renders, each of its settings unauthored there
/// taken from `settings`.
ProductSpec ReadProduct(const Scene& scene, const RenderSettingsBase& settings,
                        const RenderSettingsBase& own, const Path& path,
                        const std::vector<Path>& vars) {
    const auto pick = [&](bool authored) -> const RenderSettingsBase& {
        return authored ? own : settings;
    };
    const Path camera = pick(own.HasCamera()).GetCamera();
    const std::optional<PrimId> prim = camera.IsEmpty() ? std::nullopt : scene.Find(camera);
    if (!prim || scene.TypeName(*prim) != Camera::kName) {
        throw std::invalid_argument(Quote(path) + " names no Camera");
    }
    const glm::dvec2 resolution = pick(own.HasResolution()).GetResolution();
    const glm::uvec2 size(std::max(1.0, resolution.x), std::max(1.0, resolution.y));
    return {.product = path,
            .camera = *prim,
            .world = static_cast<std::uint32_t>(std::max(0, ReadWorld(scene, *prim))),
            .size = size,
            .lens =
                ReadLens(scene, *prim, size, pick(own.HasPixelAspectRatio()).GetPixelAspectRatio(),
                         pick(own.HasAspectRatioConformPolicy()).GetAspectRatioConformPolicy()),
            .clipping = glm::vec2(Camera(scene, camera).GetClippingRange()),
            .vars = ReadVars(scene, vars),
            .samples = static_cast<std::uint32_t>(std::max(1, pick(own.HasSamples()).GetSamples())),
            .seed = static_cast<std::uint32_t>(pick(own.HasSeed()).GetSeed()),
            .stream = HashStream(path.String())};
}

}  // namespace

std::vector<ProductSpec> ReadProducts(const Scene& scene) {
    const std::vector<Path> all = scene.Prims(RenderSettings::kName);
    if (all.empty()) {
        throw std::invalid_argument("The scene has no RenderSettings to render");
    }
    const RenderSettings settings(scene, all.front());
    const std::vector<Path> products = settings.GetProducts();
    if (products.empty()) {
        return {ReadProduct(scene, settings, settings, all.front(), {})};
    }
    std::vector<ProductSpec> specs;
    for (const Path& path : products) {
        if (!scene.Find(path) || scene.TypeName(*scene.Find(path)) != RenderProduct::kName) {
            throw std::invalid_argument(Quote(path) + " is no RenderProduct");
        }
        const RenderProduct product(scene, path);
        specs.push_back(ReadProduct(scene, settings, product, path, product.GetOrderedVars()));
    }
    return specs;
}

Lens ReadLens(const Scene& scene, PrimId prim, glm::uvec2 size, double pixelAspectRatio,
              std::string_view policy) {
    const Path path = scene.PrimPath(prim);
    if (scene.HasApi(prim, ApiName{&ErnieLensAPI::Definition()}.ToString())) {
        const ErnieLensAPI lens(scene, path);
        if (lens.GetLensProjection() != "perspective" || !lens.GetLensDistortion().empty()) {
            throw std::invalid_argument(Quote(path) + "'s fisheye and distortion models are not "
                                                      "rendered yet");
        }
    }
    const Camera camera(scene, path);
    const glm::dvec2 pixels(size);
    const glm::dvec2 film =
        ConformAperture({camera.GetHorizontalAperture(), camera.GetVerticalAperture()},
                        pixels.x * pixelAspectRatio / pixels.y, policy);
    // Pixels a metre of film, and the principal point moved against the film's offset.
    const glm::dvec2 density = pixels / film;
    const glm::dvec2 offset(camera.GetHorizontalApertureOffset(),
                            camera.GetVerticalApertureOffset());
    const glm::dvec2 principal = 0.5 * pixels + glm::dvec2(-offset.x, offset.y) * density;
    const bool orthographic = camera.GetProjection() == "orthographic";
    return {.kind = orthographic ? shaders::kOrthographic : shaders::kPinhole,
            .focal = glm::vec2(orthographic ? density : camera.GetFocalLength() * density),
            .principal = glm::vec2(principal)};
}

}  // namespace ernie

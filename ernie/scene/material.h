// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/spectrum.h"

#include <glm/glm.hpp>

#include <optional>
#include <string>
#include <vector>

namespace ernie {

/// Tiling of the detail albedo and normal textures.
struct DetailSettings {
    float tileSize =
        4.0f;  ///< World units covered by one repeat of the textures.
    float albedoStrength =
        1.0f;  ///< 0 ignores the detail albedo, 1 applies it fully.
    float normalStrength =
        1.0f;  ///< Scale on the detail normal map's slopes.
    bool directXNormals = false;  ///< Green points down the image
                                  ///< (DirectX) rather than up.
    bool hexTiling =
        true;  ///< Hide repetition with hex-tiling (Mikkelsen 2022).
    float hexRotation =
        1.0f;  ///< Random tile rotation, 0 none to 1 any angle.
    float hexContrast = 0.6f;  ///< Weight of sample luminance in the
                               ///< tile blend, 0 to 1.
    float hexSharpness =
        7.0f;  ///< Exponent on the barycentric blend weights.
};

/// How a texture repeats outside [0, 1] along one axis
/// (UsdUVTexture's wrapS and wrapT).
enum class TextureWrap { Repeat, Mirror, Clamp, Black };

/// How a texture's stored values map to linear ones (UsdUVTexture's
/// sourceColorSpace).
enum class TextureColorSpace {
    Auto,  ///< sRGB for 8-bit colour images, raw otherwise.
    Raw,
    Srgb,
};

/// Whether a texture is virtual, only the tiles in view resident
/// (`ernie:virtualTexture`).
enum class TextureVirtual {
    Auto,  ///< Albedo textures over 8192 texels a side.
    On,
    Off,
};

/// A texture read through mesh texture coordinates: a UsdUVTexture
/// with the primvar reader and UsdTransform2d feeding it, which is
/// also glTF's textureInfo with KHR_texture_transform and MaterialX's
/// image, texcoord and place2d nodes.
struct TextureReader {
    std::string file;   ///< Image file, empty when `image` names the
                        ///< texture's tensor instead.
    std::string image;  ///< Path of the Shader prim holding
                        ///< `ernie:texture`, else empty.
    std::string uvSet = "st";  ///< Mesh primvar of the coordinates
                               ///< (without "primvars:").
    int channel = -1;  ///< The output read: -1 rgb, else one channel
                       ///< (0 r to 3 a).
    TextureWrap wrapS = TextureWrap::Repeat;
    TextureWrap wrapT = TextureWrap::Repeat;
    TextureColorSpace colorSpace = TextureColorSpace::Auto;
    TextureVirtual virtualTexture = TextureVirtual::Auto;
    glm::vec4 scale{1.0f};  ///< Applied to the texture's linear
                            ///< values, then `bias` added.
    glm::vec4 bias{0.0f};
    /// UsdTransform2d of the coordinates: scaled, rotated (degrees,
    /// counter-clockwise), then translated.
    glm::vec2 uvScale{1.0f};
    float uvRotation = 0.0f;
    glm::vec2 uvTranslation{0.0f};

    /// The texture's source: its file, or the prim holding its
    /// tensor.
    const std::string& Source() const {
        return file.empty() ? image : file;
    }
};

/// How a surface reflects light: a material's settings.
struct MaterialSettings {
    std::string brdf = "lambert";  ///< Id of a model in the
                                   ///< renderer's `BrdfRegistry`.
    std::vector<float>
        brdfParams;  ///< In the model's parameter order; missing ones
                     ///< use defaults.
    glm::vec3 albedo{1.0f};  ///< Linear RGB albedo, upsampled to a
                             ///< spectrum (`RgbToSpectrum`).
    /// Reflectance spectrum replacing `albedo` when not empty.
    Spectrum reflectance;
    bool useAlbedoMap =
        true;  ///< Use the albedo map when one is set.
    DetailSettings detail;
    /// UsdPreviewSurface's diffuseColor through texture coordinates,
    /// multiplying `albedo` (white when the colour comes from the
    /// texture).
    std::optional<TextureReader> albedoTexture;
    /// UsdPreviewSurface's normal: a tangent-space normal map through
    /// texture coordinates.
    std::optional<TextureReader> normalTexture;
    /// UsdPreviewSurface's roughness from one texture channel: its
    /// alpha^2 / 2 (alpha its square) joins the pixel's unresolved
    /// slope variance, with the model's roughness then 0.
    std::optional<TextureReader> roughnessTexture;
    /// UsdPreviewSurface's metallic from one texture channel,
    /// replacing the model's parameter `metallicParam` per pixel.
    std::optional<TextureReader> metallicTexture;
    int metallicParam = -1;
    /// UsdPreviewSurface's emissiveColor: linear RGB radiance the
    /// surface emits, in cd/m^2 for white (the photometric units of
    /// lights' luminance), upsampled to a spectrum.
    glm::vec3 emission{0.0f};
    /// Its temperature in kelvin (`tokens::kThermalTemperature`):
    /// above 0 it emits emissivity times Planck's radiance in every
    /// band.
    float temperature = 0.0f;
    /// Its emissivity (`tokens::kThermalEmissivity`); empty, 1 minus
    /// its reflectance by Kirchhoff's law.
    Spectrum emissivity;
    /// Its temperature model (`tokens::kThermalModel`), an id in the
    /// renderer's `TemperatureModelRegistry`, empty for none; its
    /// parameters and data.
    std::string temperatureModel;
    std::vector<float> temperatureModelParams;
    std::vector<float> temperatureModelData;
    /// Its thermal terminal's temperature texture (`tokens::
    /// kThermalOutput`): one channel, kelvin after the reader's scale
    /// and bias, in place of the Material's temperature texture.
    std::optional<TextureReader> temperatureTexture;
    /// How much of incident polarization its diffuse reflection
    /// keeps (`tokens::kPolarizationRetention`), 0 to 1.
    float polarizationRetention = 0.0f;
    /// Its rms surface height, m (`tokens::kSpeckleRoughness`).
    float speckleRoughness = 1e-5f;
    /// UsdPreviewSurface's opacity, a constant or one texture
    /// channel, and its opacityThreshold: above 0, surfaces whose
    /// opacity falls below it are cut out (alpha testing), in shadows
    /// too. Partial opacity without a threshold (blending) is drawn
    /// opaque.
    float opacity = 1.0f;
    std::optional<TextureReader> opacityTexture;
    float opacityThreshold = 0.0f;

    /// Whether some of the surface is cut out.
    bool Cutout() const {
        return opacityThreshold > 0.0f;
    }
};

}  // namespace ernie

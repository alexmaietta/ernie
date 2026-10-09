#pragma once

#include "ernie/util/spectrum.h"

#include <glm/glm.hpp>

#include <optional>
#include <string>
#include <vector>

namespace ernie {

/// Tiling of the detail albedo and normal textures.
struct DetailSettings {
    float tileSize = 4.0f;        ///< World units covered by one repeat of the textures.
    float albedoStrength = 1.0f;  ///< 0 ignores the detail albedo, 1 applies it fully.
    float normalStrength = 1.0f;  ///< Scale on the detail normal map's slopes.
    bool directXNormals = false;  ///< Green points down the image (DirectX) rather than up.
    bool hexTiling = true;        ///< Hide repetition with hex-tiling (Mikkelsen 2022).
    float hexRotation = 1.0f;     ///< Random tile rotation, 0 none to 1 any angle.
    float hexContrast = 0.6f;     ///< Weight of sample luminance in the tile blend, 0 to 1.
    float hexSharpness = 7.0f;    ///< Exponent on the barycentric blend weights.
};

/// How a texture repeats outside [0, 1] along one axis (UsdUVTexture's wrapS and wrapT).
enum class TextureWrap { Repeat, Mirror, Clamp, Black };

/// How a texture's stored values map to linear ones (UsdUVTexture's sourceColorSpace).
enum class TextureColorSpace {
    Auto,  ///< sRGB for 8-bit colour images, raw otherwise.
    Raw,
    Srgb,
};

/// A texture read through mesh texture coordinates: a UsdUVTexture with the primvar reader and
/// UsdTransform2d feeding it, which is also glTF's textureInfo with KHR_texture_transform and
/// MaterialX's image, texcoord and place2d nodes.
struct TextureReader {
    std::string file;  ///< Image file, empty when `image` names the texture's tensor instead.
    std::string image;  ///< Path of the Shader prim holding `ernie:texture`, else empty.
    std::string uvSet = "st";  ///< Mesh primvar of the coordinates (without "primvars:").
    int channel = -1;  ///< The output read: -1 rgb, else one channel (0 r to 3 a).
    TextureWrap wrapS = TextureWrap::Repeat;
    TextureWrap wrapT = TextureWrap::Repeat;
    TextureColorSpace colorSpace = TextureColorSpace::Auto;
    glm::vec4 scale{1.0f};  ///< Applied to the texture's linear values, then `bias` added.
    glm::vec4 bias{0.0f};
    /// UsdTransform2d of the coordinates: scaled, rotated (degrees, counter-clockwise), then
    /// translated.
    glm::vec2 uvScale{1.0f};
    float uvRotation = 0.0f;
    glm::vec2 uvTranslation{0.0f};

    /// The texture's source: its file, or the prim holding its tensor.
    const std::string& Source() const {
        return file.empty() ? image : file;
    }
};

/// How a surface reflects light: a material's settings.
struct MaterialSettings {
    std::string brdf = "lambert";   ///< Id of a model in the renderer's `BrdfRegistry`.
    std::vector<float> brdfParams;  ///< In the model's parameter order; missing ones use defaults.
    glm::vec3 albedo{1.0f};  ///< Linear RGB albedo, upsampled to a spectrum (`RgbToSpectrum`).
    /// Reflectance spectrum replacing `albedo` when not empty.
    Spectrum reflectance;
    bool useAlbedoMap = true;  ///< Use the albedo map when one is set.
    DetailSettings detail;
    /// UsdPreviewSurface's diffuseColor through texture coordinates, multiplying `albedo` (white
    /// when the colour comes from the texture).
    std::optional<TextureReader> albedoTexture;
    /// UsdPreviewSurface's normal: a tangent-space normal map through texture coordinates.
    std::optional<TextureReader> normalTexture;
    /// UsdPreviewSurface's roughness from one texture channel: its alpha^2 / 2 (alpha its
    /// square) joins the pixel's unresolved slope variance, with the model's roughness then 0.
    std::optional<TextureReader> roughnessTexture;
    /// UsdPreviewSurface's metallic from one texture channel, replacing the model's parameter
    /// `metallicParam` per pixel.
    std::optional<TextureReader> metallicTexture;
    int metallicParam = -1;
};

}  // namespace ernie

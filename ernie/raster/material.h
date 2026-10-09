#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/frame_uploads.h"
#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/scene/material.h"
#include "ernie/util/image.h"
#include "ernie/util/raster_view.h"
#include "ernie/util/spectrum.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ernie {

/// A material's GPU resources: the material uniform, an optional albedo map and optional tileable
/// detail albedo and normal textures, bound at bindings 5 to 9 of every raster pass
/// (`surface.slang`), so terrain and meshes share materials.
///
/// The albedo map is a georeferenced raster sampled by world x and y, like the heightfield but
/// with its own grid. Detail textures repeat every `DetailSettings::tileSize` world units: the
/// terrain projects them from above and meshes along all three axes (triplanar); the
/// detail albedo multiplies the base albedo after division by its own mean, so it adds variation
/// without changing the map's calibrated mean, and the detail normal map's slopes add to the
/// terrain's. Detail textures are mipmapped and sampled anisotropically.
///
/// References:
/// - M. S. Mikkelsen, "Practical Real-Time Hex-Tiling", Journal of Computer Graphics Techniques
///   11(3), 2022.
/// - M. Jarzynski and M. Olano, "Hash Functions for GPU Rendering", Journal of Computer Graphics
///   Techniques 9(3), 2020 (pcg2d, for per-tile rotations and offsets).
class RasterMaterial {
  public:
    /// Its uniforms reach the device through `uploads`, which must outlive it.
    RasterMaterial(const VulkanContext& context, FrameUploads& uploads);
    RasterMaterial(const RasterMaterial&) = delete;
    RasterMaterial& operator=(const RasterMaterial&) = delete;

    /// The material's uniform buffer, albedo map, detail albedo, detail normal and detail sampler,
    /// in binding order (5 to 9).
    void AppendResources(std::vector<DescriptorResource>& resources) const;

    /// Uploads `albedo` as the albedo map; the data is copied, so the view need not outlive the
    /// call. Cells that are not finite or equal the nodata value fall back to
    /// `MaterialSettings::albedo`.
    /// @throws std::invalid_argument if the view is empty or malformed (see `RasterView`), or
    /// larger than the device's maximum texture size.
    void SetAlbedoMap(const RasterView& albedo);
    void ClearAlbedoMap();

    bool HasAlbedoMap() const {
        return hasAlbedoMap_;
    }

    /// Uploads detail textures; nullptr removes one. `albedo` holds linear colour (load it with
    /// `ImageEncoding::Srgb`); `normal` holds tangent-space normals encoded as n / 2 + 1/2 (load
    /// with `ImageEncoding::Linear`), x east and y up the image.
    /// @throws std::invalid_argument if an image is empty or larger than the device's maximum
    /// texture size.
    void SetDetailTextures(const Image* albedo, const Image* normal);

    /// Replaces only the detail albedo texture; nullptr removes it.
    /// @throws std::invalid_argument as `SetDetailTextures`.
    void SetDetailAlbedo(const Image* albedo);

    /// Replaces only the detail normal map; nullptr removes it.
    /// @throws std::invalid_argument as `SetDetailTextures`.
    void SetDetailNormal(const Image* normal);

    /// Uploads the texture of `MaterialSettings::albedoTexture` (linear colour), sampled through
    /// mesh texture coordinates as `reader` wraps them; nullptr removes it. Its scale, bias and
    /// coordinate transform come from the settings of `Update`.
    /// @throws std::invalid_argument if the image is empty or larger than the device allows.
    void SetAlbedoTexture(const Image* albedo, const TextureReader& reader);

    /// Uploads the tangent-space normal map of `MaterialSettings::normalTexture`: `normal` holds
    /// raw values, which `reader`'s scale and bias turn into normals (UsdPreviewSurface's
    /// convention; scale 2 and bias -1 for the usual n / 2 + 1/2 encoding). nullptr removes it.
    /// @throws std::invalid_argument as `SetAlbedoTexture`.
    void SetNormalTexture(const Image* normal, const TextureReader& reader);

    /// Uploads the one-channel textures of `MaterialSettings::roughnessTexture` (`metallic`
    /// false) or `metallicTexture` (true): `reader`'s channel of `image`, scaled and biased as it
    /// says, clamped to [0, 1]. nullptr removes it.
    /// @throws std::invalid_argument as `SetAlbedoTexture`.
    void SetScalarTexture(bool metallic, const Image* image, const TextureReader& reader);

    bool HasAlbedoTexture() const {
        return hasAlbedoTexture_;
    }

    bool HasRoughnessTexture() const {
        return hasScalarTexture_[0];
    }

    bool HasMetallicTexture() const {
        return hasScalarTexture_[1];
    }

    bool HasNormalTexture() const {
        return hasNormalTexture_;
    }

    /// The textures mesh passes read through texture coordinates, albedo, normal map, roughness
    /// and metallic, each with its own sampler (bindings 22 to 25 of mesh.slang).
    void AppendTextureResources(std::vector<DescriptorResource>& resources) const;

    bool HasDetailAlbedo() const {
        return hasDetailAlbedo_;
    }

    bool HasDetailNormal() const {
        return hasDetailNormal_;
    }

    /// Mean linear colour of the detail albedo texture as stored (after 8-bit sRGB quantisation),
    /// (1, 1, 1) without one.
    glm::vec3 DetailAlbedoMean() const {
        return detailMean_;
    }

    /// Sets the material's settings; `brdfParams` are the model's resolved values. This and the
    /// texture setters write the material's uniform, so call them only while no submitted work is
    /// reading the material.
    void Update(const MaterialSettings& settings, std::span<const float> brdfParams);

    /// Moves the origin positions are relative to (`FrameParams::origin`): the albedo map's
    /// placement and the detail tiling's phase follow, so both stay fixed to the world.
    void SetOrigin(const glm::dvec3& origin);

    /// The settings of the last `Update`.
    const MaterialSettings& Settings() const {
        return settings_;
    }

    /// The model's resolved parameters from the last `Update`.
    const std::vector<float>& Params() const {
        return params_;
    }

    /// The material's reflectance spectrum: `Settings().reflectance`, or the albedo upsampled.
    const Spectrum& Reflectance() const {
        return reflectance_;
    }

    /// Luminance of the reflectance (CIE Y under the equal-energy illuminant), which albedo map
    /// values are relative to: the map sets the albedo and the spectrum its shape.
    float Luminance() const {
        return luminance_;
    }

    /// GPU memory held for textures and the uniform, in bytes.
    std::uint64_t GpuBytes() const;

  private:
    void CheckDetailImage(const Image* image) const;
    void WriteUniform();
    std::unique_ptr<GpuImage> UploadMips(const Image& image, MipFilter filter,
                                         ImageEncoding encoding, VkFormat format) const;

    const VulkanContext& context_;
    FrameUploads& uploads_;
    MaterialSettings settings_;
    std::vector<float> params_;
    Spectrum reflectance_ = Spectrum::Constant(1.0);
    float luminance_ = 1.0f;
    GpuBuffer uniform_;
    Sampler sampler_;
    std::unique_ptr<GpuImage> albedoMap_;
    std::unique_ptr<GpuImage> detailAlbedo_;
    std::unique_ptr<GpuImage> detailNormal_;
    std::unique_ptr<GpuImage> albedoTexture_;
    std::unique_ptr<GpuImage> normalTexture_;
    std::unique_ptr<Sampler> albedoSampler_;
    std::unique_ptr<Sampler> normalSampler_;
    std::array<std::unique_ptr<GpuImage>, 2> scalarTextures_;  ///< Roughness, metallic.
    std::array<std::unique_ptr<Sampler>, 2> scalarSamplers_;

    bool hasAlbedoMap_ = false;
    bool hasAlbedoTexture_ = false;
    bool hasNormalTexture_ = false;
    std::array<bool, 2> hasScalarTexture_{false, false};
    bool hasDetailAlbedo_ = false;
    bool hasDetailNormal_ = false;
    glm::vec3 detailMean_{1.0f};
    glm::dvec4 albedoGrid_{0.0};  ///< Origin x, origin y, pixel size, unused.
    glm::dvec3 origin_{0.0};      ///< The frame's origin (`SetOrigin`).
    glm::uvec2 albedoSize_{1u, 1u};
};

}  // namespace ernie

// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/bindless.h"
#include "ernie/device/buffer.h"
#include "ernie/device/frame_uploads.h"
#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/raster/virtual_texture.h"
#include "ernie/scene/material.h"
#include "ernie/util/image.h"
#include "ernie/util/raster_view.h"
#include "ernie/util/spectrum.h"
#include "ernie/util/texture_file.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ernie {

/// The textures a UsdPreviewSurface reads through mesh texture
/// coordinates (bindings 22 to 26 of mesh.slang, in this order).
enum class MaterialTextureSlot : std::uint32_t {
    Albedo,
    Normal,
    Roughness,
    Metallic,
    Opacity,
};

inline constexpr std::size_t kMaterialTextureSlots = 5;

/// A texture uploaded for one `MaterialTextureSlot`, which every
/// material reading the same source the same way can share
/// (`RasterDelegate` keeps a cache of them).
struct MaterialTexture {
    /// The texture; null when it is virtual.
    std::unique_ptr<GpuImage> image;
    /// A virtual texture (`TextureReader::virtualTexture`): only the
    /// tiles in view resident.
    std::shared_ptr<VirtualTexture> virtualTexture;
    /// For normal maps, how mesh.slang decodes it
    /// (`Surface::textures.y`): 1 n * 127 + 128 with the slope
    /// deviation in alpha (uploaded from an `Image`), 2 n / 2 + 1/2
    /// in rgb, 3 the same in rg with z rebuilt (stored textures).
    float normalEncoding = 1.0f;
};

/// Uploads `image` as `slot`'s texture as `reader` reads it, mipped:
/// albedo as linear colour (its scale and bias applied by the
/// shader); a normal map's raw values turned into normals by
/// `reader`'s scale and bias (UsdPreviewSurface's convention; scale 2
/// and bias -1 for the usual n / 2 + 1/2 encoding); one-channel
/// textures `reader`'s channel, scaled and biased as it says, clamped
/// to [0, 1].
/// @throws std::invalid_argument if the image is empty or larger than
/// the device allows.
std::shared_ptr<const MaterialTexture>
UploadMaterialTexture(const VulkanContext& context,
                      MaterialTextureSlot slot, const Image& image,
                      const TextureReader& reader);

/// As above, for a texture read as stored from a DDS or KTX2 file
/// (`LoadTextureFile`): the first layer's mips upload as they are,
/// BC1 to BC7 blocks included, decoded (`DecodeBlocks`) only on
/// devices without `VulkanContext::SupportsTextureCompressionBc`. A
/// reader's colour space
/// picks the sRGB or plain variant of formats with both (raw plain,
/// sRGB sRGB, auto the file's own); one- and two-channel albedo reads
/// as grey (and alpha). One-channel textures take `reader`'s channel
/// through the image view, with no scale or bias. Normal maps hold n
/// / 2 + 1/2 under the usual scale 2 and bias -1, in three channels
/// or in two (BC5), the shader rebuilding z; they carry no slope
/// deviation for roughness, as `MipFilter::NormalSlopes` adds.
///
/// Albedo textures become virtual (`VirtualTexture`, keeping
/// `texture`) when the reader says so, or by default when a side is
/// over `kVirtualTextureSide` texels, on devices with sparse
/// residency; they then need every mip.
/// @throws std::invalid_argument for a texture larger than the device
/// allows, BC2, BC6H or signed BC4 or BC5 on a device that cannot
/// sample them (`DecodeBlocks`), a scale or bias
/// other than those, a normal map of one channel, or a virtual
/// texture without every mip; std::runtime_error for a virtual
/// texture the device cannot make.
std::shared_ptr<const MaterialTexture>
UploadMaterialTexture(const VulkanContext& context,
                      MaterialTextureSlot slot, TextureData texture,
                      const TextureReader& reader);

/// Albedo textures with a side over this many texels are virtual
/// unless their reader says otherwise.
inline constexpr std::uint32_t kVirtualTextureSide = 8192;

/// A material's GPU resources: the material uniform, an optional
/// albedo map and optional tileable detail albedo and normal
/// textures, its uniform bound at binding 5 of every raster pass
/// (`surface.slang`) and naming the textures by their bindless heap
/// slots, so terrain and meshes share materials.
///
/// The albedo map is a georeferenced raster sampled by world x and y,
/// like the heightfield but with its own grid. Detail textures repeat
/// every `DetailSettings::tileSize` world units: the terrain projects
/// them from above and meshes along all three axes (triplanar); the
/// detail albedo multiplies the base albedo after division by its own
/// mean, so it adds variation without changing the map's calibrated
/// mean, and the detail normal map's slopes add to the terrain's.
/// Detail textures are mipmapped and sampled anisotropically.
///
/// References:
/// - M. S. Mikkelsen, "Practical Real-Time Hex-Tiling", Journal of
///   Computer Graphics Techniques 11(3), 2022.
/// - M. Jarzynski and M. Olano, "Hash Functions for GPU Rendering",
///   Journal of Computer Graphics Techniques 9(3), 2020 (pcg2d, for
///   per-tile rotations and offsets).
class RasterMaterial {
  public:
    /// Its uniforms reach the device through `uploads`, which must
    /// outlive it.
    RasterMaterial(const VulkanContext& context,
                   FrameUploads& uploads);
    RasterMaterial(const RasterMaterial&) = delete;
    RasterMaterial& operator=(const RasterMaterial&) = delete;

    /// The material's uniform buffer (binding 5), which names its
    /// albedo map, detail textures and sampler, and the textures mesh
    /// passes read through texture coordinates (albedo, normal map,
    /// roughness, metallic and opacity, each with its own sampler) by
    /// their slots in the bindless heap.
    void
    AppendResources(std::vector<DescriptorResource>& resources) const;

    /// Uploads `albedo` as the albedo map; the data is copied, so the
    /// view need not outlive the call. Cells that are not finite or
    /// equal the nodata value fall back to
    /// `MaterialSettings::albedo`.
    /// @throws std::invalid_argument if the view is empty or
    /// malformed (see `RasterView`), or larger than the device's
    /// maximum texture size.
    void SetAlbedoMap(const RasterView& albedo);
    void ClearAlbedoMap();

    bool HasAlbedoMap() const {
        return hasAlbedoMap_;
    }

    /// Uploads detail textures; nullptr removes one. `albedo` holds
    /// linear colour (load it with `ImageEncoding::Srgb`); `normal`
    /// holds tangent-space normals encoded as n / 2 + 1/2 (load with
    /// `ImageEncoding::Linear`), x east and y up the image.
    /// @throws std::invalid_argument if an image is empty or larger
    /// than the device's maximum texture size.
    void SetDetailTextures(const Image* albedo, const Image* normal);

    /// Replaces only the detail albedo texture; nullptr removes it.
    /// @throws std::invalid_argument as `SetDetailTextures`.
    void SetDetailAlbedo(const Image* albedo);

    /// Replaces only the detail normal map; nullptr removes it.
    /// @throws std::invalid_argument as `SetDetailTextures`.
    void SetDetailNormal(const Image* normal);

    /// Replaces the temperature texture: `kelvin`, `width` x `height`
    /// row by row from the top, read through mesh texture coordinates
    /// with thermal emission on (bilinear, clamped at its edges);
    /// empty removes it.
    /// @throws std::invalid_argument if the size does not match or
    /// exceeds the device's maximum texture size.
    void SetTemperatureTexture(std::span<const float> kelvin,
                               std::uint32_t width,
                               std::uint32_t height);

    bool HasTemperatureTexture() const {
        return temperatureTexture_ != nullptr;
    }

    /// Reads `texture` (from `UploadMaterialTexture`, and shareable
    /// with other materials) for `slot` through mesh texture
    /// coordinates, wrapping as `reader` says; nullptr removes it.
    /// Its scale, bias and coordinate transform come from the
    /// settings of `Update`.
    void SetTexture(MaterialTextureSlot slot,
                    std::shared_ptr<const MaterialTexture> texture,
                    const TextureReader& reader);

    bool HasTexture(MaterialTextureSlot slot) const {
        return textures_[static_cast<std::size_t>(slot)] != nullptr;
    }

    bool HasDetailAlbedo() const {
        return hasDetailAlbedo_;
    }

    bool HasDetailNormal() const {
        return hasDetailNormal_;
    }

    /// Mean linear colour of the detail albedo texture as stored
    /// (after 8-bit sRGB quantisation), (1, 1, 1) without one.
    glm::vec3 DetailAlbedoMean() const {
        return detailMean_;
    }

    /// Sets the material's settings; `brdfParams` are the model's
    /// resolved values, and `gloss` its roughness and metallic for
    /// minimal shading's glossy highlights (`MinimalMode::Glossy`),
    /// none for a model without a roughness. This and the texture
    /// setters write the material's uniform, so call them only while
    /// no submitted work is reading the material.
    void Update(const MaterialSettings& settings,
                std::span<const float> brdfParams,
                std::optional<glm::vec2> gloss = std::nullopt);

    /// Moves the origin positions are relative to
    /// (`FrameParams::origin`): the albedo map's placement and the
    /// detail tiling's phase follow, so both stay fixed to the world.
    void SetOrigin(const glm::dvec3& origin);

    /// Begins frame `frame` in frame slot `slot` for its virtual
    /// textures: serves the tiles their last frame in the slot asked
    /// for (`VirtualTexture::BeginFrame`) and points the uniform at
    /// the slot's request flags. Nothing without virtual textures.
    void BeginFrame(std::uint32_t slot, std::uint64_t frame);

    /// The settings of the last `Update`.
    const MaterialSettings& Settings() const {
        return settings_;
    }

    /// The model's resolved parameters from the last `Update`.
    const std::vector<float>& Params() const {
        return params_;
    }

    /// The material's reflectance spectrum: `Settings().reflectance`,
    /// or the albedo upsampled.
    const Spectrum& Reflectance() const {
        return reflectance_;
    }

    /// Luminance of the reflectance (CIE Y under the equal-energy
    /// illuminant), which albedo map values are relative to: the map
    /// sets the albedo and the spectrum its shape.
    float Luminance() const {
        return luminance_;
    }

    /// GPU memory held for the uniform and the material's own
    /// textures, in bytes: not those of `SetTexture`, which materials
    /// may share.
    std::uint64_t GpuBytes() const;

  private:
    void WriteUniform();

    const VulkanContext& context_;
    FrameUploads& uploads_;
    MaterialSettings settings_;
    std::vector<float> params_;
    std::optional<glm::vec2> gloss_;  ///< `Update`'s.
    Spectrum reflectance_ = Spectrum::Constant(1.0);
    float luminance_ = 1.0f;
    GpuBuffer uniform_;
    Sampler sampler_;
    std::unique_ptr<GpuImage> albedoMap_;
    std::unique_ptr<GpuImage> detailAlbedo_;
    std::unique_ptr<GpuImage> detailNormal_;
    std::unique_ptr<GpuImage> temperatureTexture_;  ///< R32F kelvin.
    /// By `MaterialTextureSlot`: the texture of `SetTexture`, null
    /// for none; the 1 x 1 one bound in its place (white albedo, a
    /// flat normal, zero); and the sampler.
    std::array<std::shared_ptr<const MaterialTexture>,
               kMaterialTextureSlots>
        textures_;
    std::array<std::unique_ptr<GpuImage>, kMaterialTextureSlots>
        fallbacks_;
    std::array<std::unique_ptr<Sampler>, kMaterialTextureSlots>
        samplers_;
    /// The textures' and samplers' slots in the bindless heap that
    /// the uniform names, as of its last write (`WriteUniform`).
    std::vector<BindlessHeap::Slot> heapSlots_;

    bool hasAlbedoMap_ = false;
    bool hasDetailAlbedo_ = false;
    bool hasDetailNormal_ = false;
    glm::vec3 detailMean_{1.0f};
    glm::dvec4 albedoGrid_{
        0.0};  ///< Origin x, origin y, pixel size, unused.
    glm::dvec3 origin_{0.0};  ///< The frame's origin (`SetOrigin`).
    std::uint32_t frameSlot_ = 0;  ///< The last `BeginFrame`'s slot.
    glm::uvec2 albedoSize_{1u, 1u};
};

}  // namespace ernie

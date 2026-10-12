// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/material.h"

#include "ernie/scene/brdf.h"
#include "ernie/util/raster_checks.h"

#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <stdexcept>

namespace ernie {

namespace {

struct SurfaceUniforms {
    float albedo[4];
    float albedoMap[4];
    float albedoMapSize[4];
    float detail[4];
    float detailMean[4];
    float hex[4];
    float detailPhase[4];
    float hexPhaseXyYz[4];
    float hexPhaseXz[4];
    float albedoTextureScale[4];
    float albedoTextureBias[4];
    float albedoUv[8];  ///< Rows of the 2 x 3 texture coordinate
                        ///< transform.
    float normalUv[8];
    float roughnessUv[8];
    float metallicUv[8];
    float textures[4];  ///< Has albedo texture, normal map,
                        ///< roughness, metallic.
    float opacityUv[8];
    float opacity[4];  ///< Opacity, its cutout threshold, has a
                       ///< texture, unused.
    float params[BrdfRegistry::kMaxParams];
    /// Minimal shading's diffuse, glossy and emission mode: the
    /// model's roughness and metallic and 1 when it has a roughness;
    /// the emission, linear RGB cd/m^2.
    float gloss[4];
    float emission[4];
    /// Bindless heap slots: the albedo map, detail albedo and detail
    /// normal textures, the detail sampler; the mesh textures'
    /// images, then their samplers, by `MaterialTextureSlot`.
    std::uint32_t heap[16];
    /// A virtual albedo texture (`VirtualTexture`): its first tail
    /// mip, mips (0 for none), width and height; texels across and
    /// down a tile; the address of its frame slot's request flags.
    std::uint32_t virtualAlbedo[4];
    std::uint32_t virtualTile[2];
    std::uint64_t virtualRequests;
};
static_assert(sizeof(SurfaceUniforms) == 560,
              "SurfaceUniforms must match Surface in surface.slang");

/// UsdTransform2d as the rows of a 2 x 3 matrix: scale, then rotate
/// counter-clockwise, then translate.
void WriteUvTransform(const TextureReader& reader, float (&rows)[8]) {
    const float angle = glm::radians(reader.uvRotation);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const glm::vec2 k = reader.uvScale;
    const float values[8] = {
        c * k.x, -s * k.y, reader.uvTranslation.x, 0.0f,
        s * k.x, c * k.y,  reader.uvTranslation.y, 0.0f};
    std::copy(std::begin(values), std::end(values), rows);
}

VkSamplerAddressMode AddressModeOf(TextureWrap wrap) {
    switch (wrap) {
    case TextureWrap::Mirror:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case TextureWrap::Clamp:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case TextureWrap::Black:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case TextureWrap::Repeat:
        break;
    }
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkFormat VkFormatOf(TextureFormat format) {
    using F = TextureFormat;
    switch (format) {
    case F::R8:
        return VK_FORMAT_R8_UNORM;
    case F::Rg8:
        return VK_FORMAT_R8G8_UNORM;
    case F::Rgba8:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case F::Rgba8Srgb:
        return VK_FORMAT_R8G8B8A8_SRGB;
    case F::Bgra8:
        return VK_FORMAT_B8G8R8A8_UNORM;
    case F::Bgra8Srgb:
        return VK_FORMAT_B8G8R8A8_SRGB;
    case F::Rgba16F:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case F::Rgba32F:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    case F::Bc1:
        return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case F::Bc1Srgb:
        return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    case F::Bc2:
        return VK_FORMAT_BC2_UNORM_BLOCK;
    case F::Bc2Srgb:
        return VK_FORMAT_BC2_SRGB_BLOCK;
    case F::Bc3:
        return VK_FORMAT_BC3_UNORM_BLOCK;
    case F::Bc3Srgb:
        return VK_FORMAT_BC3_SRGB_BLOCK;
    case F::Bc4:
        return VK_FORMAT_BC4_UNORM_BLOCK;
    case F::Bc4Snorm:
        return VK_FORMAT_BC4_SNORM_BLOCK;
    case F::Bc5:
        return VK_FORMAT_BC5_UNORM_BLOCK;
    case F::Bc5Snorm:
        return VK_FORMAT_BC5_SNORM_BLOCK;
    case F::Bc6hUfloat:
        return VK_FORMAT_BC6H_UFLOAT_BLOCK;
    case F::Bc6hSfloat:
        return VK_FORMAT_BC6H_SFLOAT_BLOCK;
    case F::Bc7:
        return VK_FORMAT_BC7_UNORM_BLOCK;
    case F::Bc7Srgb:
        return VK_FORMAT_BC7_SRGB_BLOCK;
    }
    throw std::invalid_argument("Unknown texture format");
}

/// A trilinear, anisotropic sampler wrapping as `reader` does.
std::unique_ptr<Sampler> TextureSampler(const VulkanContext& context,
                                        const TextureReader& reader) {
    SamplerDesc desc{VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                     AddressModeOf(reader.wrapS), 16.0f, false};
    desc.addressModeV = AddressModeOf(reader.wrapT);
    desc.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    return std::make_unique<Sampler>(context, desc);
}

/// surface.slang's hex lattice: its skew (tile coordinates times
/// `kSkewScale`, then x minus `kSkewX` times y and y times `kSkewY`)
/// and the period of its random rotations.
constexpr double kSkewScale = 3.4641016151377544;  // 2 sqrt(3)
constexpr double kSkewX = 0.57735027;
constexpr double kSkewY = 1.15470054;
constexpr double kHexPeriod = 4096.0;

/// The hex lattice's phase at world plane point `planar` (in tiles):
/// the tile offset that puts the origin-relative lattice where the
/// world's is, modulo the rotations' period.
glm::dvec2 HexPhase(const glm::dvec2& planar) {
    const glm::dvec2 st = planar * kSkewScale;
    glm::dvec2 skewed(st.x - kSkewX * st.y, kSkewY * st.y);
    skewed -= kHexPeriod * glm::floor(skewed / kHexPeriod);
    const double y = skewed.y / (kSkewScale * kSkewY);
    return {skewed.x / kSkewScale + kSkewX * y, y};
}

glm::vec3 QuantisedSrgbMean(const Image& image) {
    const std::vector<std::uint8_t> bytes =
        ToRgba8(image, ImageEncoding::Srgb);
    glm::dvec3 sum(0.0);
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        for (int c = 0; c < 3; ++c) {
            const float stored = static_cast<float>(
                bytes[4 * i + static_cast<std::size_t>(c)]);
            sum[c] +=
                static_cast<double>(SrgbToLinear(stored / 255.0f));
        }
    }
    return glm::vec3(sum / static_cast<double>(image.pixels.size()));
}

std::span<const std::byte> Bytes(const auto& values) {
    return std::as_bytes(std::span(values));
}

/// Throws unless `image` (when given) has pixels and fits the device.
void CheckImage(const VulkanContext& context, const Image* image) {
    if (image == nullptr) {
        return;
    }
    if (image->width == 0 || image->height == 0 ||
        image->pixels.size() !=
            static_cast<std::size_t>(image->width) * image->height) {
        throw std::invalid_argument("Texture is empty");
    }
    const std::uint32_t maxSize =
        context.Limits().maxImageDimension2D;
    if (image->width > maxSize || image->height > maxSize) {
        throw std::invalid_argument(
            "Texture exceeds the device's maximum texture size");
    }
}

/// `image` and its mips (`MipChain`) as 8-bit texels of `format`.
std::unique_ptr<GpuImage> UploadMips(const VulkanContext& context,
                                     const Image& image,
                                     MipFilter filter,
                                     ImageEncoding encoding,
                                     VkFormat format) {
    std::vector<Image> mips = MipChain(image, filter);
    if (filter == MipFilter::NormalSlopes) {
        // Normals stored as n * 127 + 128 in 8 bits, so flat is
        // exactly byte 128 and decodes to exactly 0 (surface.slang's
        // `detailTap`); n / 2 + 1/2 would store 0 as 127.5, tilting
        // every normal by 1/255. Source maps that stored flat as 128
        // decode to flat too.
        for (Image& mip : mips) {
            for (glm::vec4& texel : mip.pixels) {
                const glm::vec3 n = 2.0f * glm::vec3(texel) - 1.0f;
                const glm::vec3 bytes = glm::round(
                    glm::clamp(n * 127.0f + 128.0f, 1.0f, 255.0f));
                texel = glm::vec4(bytes / 255.0f, texel.a);
            }
        }
    }
    auto texture = std::make_unique<GpuImage>(
        context, ImageDesc{image.width, image.height, 1,
                           static_cast<std::uint32_t>(mips.size()),
                           format, VK_IMAGE_USAGE_SAMPLED_BIT});
    std::vector<std::vector<std::uint8_t>> data;
    std::vector<std::span<const std::byte>> spans;
    for (const Image& mip : mips) {
        data.push_back(ToRgba8(mip, encoding));
    }
    for (const auto& bytes : data) {
        spans.push_back(Bytes(bytes));
    }
    texture->Upload(spans);
    return texture;
}

/// A stored texture's format as the device holds it: its own, or its
/// sRGB or plain variant per `reader`.
/// @throws std::invalid_argument for BC blocks the device cannot
/// sample.
VkFormat StoredFormat(const VulkanContext& context,
                      const TextureData& texture,
                      const TextureReader& reader) {
    TextureFormat format = texture.format;
    if (reader.colorSpace != TextureColorSpace::Auto) {
        format = WithSrgb(format, reader.colorSpace ==
                                      TextureColorSpace::Srgb);
    }
    if (InfoOf(format).compressed() &&
        !context.SupportsTextureCompressionBc()) {
        throw std::invalid_argument(
            "The device cannot sample BC-compressed textures");
    }
    return VkFormatOf(format);
}

/// The first layer's mips of a stored texture, in its format or that
/// format's sRGB or plain variant per `reader`, viewed through
/// `components`.
std::unique_ptr<GpuImage>
UploadStored(const VulkanContext& context, const TextureData& texture,
             const TextureReader& reader,
             VkComponentMapping components) {
    const VkFormat format = StoredFormat(context, texture, reader);
    const std::uint32_t maxSize =
        context.Limits().maxImageDimension2D;
    if (texture.width > maxSize || texture.height > maxSize) {
        throw std::invalid_argument(
            "Texture exceeds the device's maximum texture size");
    }
    ImageDesc desc{
        texture.width, texture.height, 1,
        texture.mips,  format,         VK_IMAGE_USAGE_SAMPLED_BIT};
    desc.components = components;
    auto image = std::make_unique<GpuImage>(context, desc);
    std::vector<std::span<const std::byte>> mips;
    for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
        mips.push_back(texture.Mip(0, mip));
    }
    image->Upload(mips);
    return image;
}

/// What a slot reads without a texture: white albedo, a flat normal,
/// zero.
std::unique_ptr<GpuImage>
FallbackTexture(const VulkanContext& context,
                MaterialTextureSlot slot) {
    switch (slot) {
    case MaterialTextureSlot::Albedo:
        return UploadMips(context, Image(1, 1, glm::vec4(1.0f)),
                          MipFilter::Average, ImageEncoding::Srgb,
                          VK_FORMAT_R8G8B8A8_SRGB);
    case MaterialTextureSlot::Normal:
        return UploadMips(
            context, Image(1, 1, glm::vec4(0.5f, 0.5f, 1.0f, 1.0f)),
            MipFilter::NormalSlopes, ImageEncoding::Linear,
            VK_FORMAT_R8G8B8A8_UNORM);
    default:
        return UploadMips(context, Image(1, 1, glm::vec4(0.0f)),
                          MipFilter::Average, ImageEncoding::Linear,
                          VK_FORMAT_R8G8B8A8_UNORM);
    }
}

}  // namespace

std::shared_ptr<const MaterialTexture>
UploadMaterialTexture(const VulkanContext& context,
                      MaterialTextureSlot slot, const Image& image,
                      const TextureReader& reader) {
    CheckImage(context, &image);
    auto texture = std::make_shared<MaterialTexture>();
    if (slot == MaterialTextureSlot::Albedo) {
        texture->image =
            UploadMips(context, image, MipFilter::Average,
                       ImageEncoding::Srgb, VK_FORMAT_R8G8B8A8_SRGB);
    } else if (slot == MaterialTextureSlot::Normal) {
        // The reader's scale and bias turn the raw values into
        // normals, stored as n / 2 + 1/2 for
        // `MipFilter::NormalSlopes`.
        Image normals = image;
        for (glm::vec4& texel : normals.pixels) {
            glm::vec3 n = glm::vec3(texel) * glm::vec3(reader.scale) +
                          glm::vec3(reader.bias);
            n = glm::length(n) > 0.0f ? glm::normalize(n)
                                      : glm::vec3(0.0f, 0.0f, 1.0f);
            texel = glm::vec4(0.5f * n + 0.5f, 0.0f);
        }
        texture->image = UploadMips(
            context, normals, MipFilter::NormalSlopes,
            ImageEncoding::Linear, VK_FORMAT_R8G8B8A8_UNORM);
    } else {
        Image values = image;
        const int channel = std::clamp(reader.channel, 0, 3);
        for (glm::vec4& texel : values.pixels) {
            const float value =
                std::clamp(texel[channel] * reader.scale[channel] +
                               reader.bias[channel],
                           0.0f, 1.0f);
            texel = glm::vec4(value, value, value, 1.0f);
        }
        texture->image = UploadMips(
            context, values, MipFilter::Average,
            ImageEncoding::Linear, VK_FORMAT_R8G8B8A8_UNORM);
    }
    return texture;
}

std::shared_ptr<const MaterialTexture>
UploadMaterialTexture(const VulkanContext& context,
                      MaterialTextureSlot slot, TextureData stored,
                      const TextureReader& reader) {
    auto texture = std::make_shared<MaterialTexture>();
    const std::uint32_t channels = InfoOf(stored.format).channels;
    if (slot == MaterialTextureSlot::Albedo) {
        // Grey textures, with alpha in a second channel, read as rgb.
        VkComponentMapping components{};
        if (channels <= 2) {
            components = {VK_COMPONENT_SWIZZLE_R,
                          VK_COMPONENT_SWIZZLE_R,
                          VK_COMPONENT_SWIZZLE_R,
                          channels == 2 ? VK_COMPONENT_SWIZZLE_G
                                        : VK_COMPONENT_SWIZZLE_ONE};
        }
        const bool large = std::max(stored.width, stored.height) >
                           kVirtualTextureSide;
        if (reader.virtualTexture == TextureVirtual::On ||
            (reader.virtualTexture == TextureVirtual::Auto && large &&
             context.SupportsSparseResidency())) {
            const VkFormat format =
                StoredFormat(context, stored, reader);
            texture->virtualTexture =
                std::make_shared<VirtualTexture>(
                    context,
                    std::make_shared<const TextureData>(
                        std::move(stored)),
                    format, components);
        } else {
            texture->image =
                UploadStored(context, stored, reader, components);
        }
    } else if (slot == MaterialTextureSlot::Normal) {
        if (channels < 2) {
            throw std::invalid_argument(
                "A normal map needs two or three channels");
        }
        const int axes = channels == 2 ? 2 : 3;
        for (int c = 0; c < axes; ++c) {
            if (reader.scale[c] != 2.0f || reader.bias[c] != -1.0f) {
                throw std::invalid_argument(
                    "A stored normal map is read with scale 2 and "
                    "bias "
                    "-1");
            }
        }
        texture->image = UploadStored(context, stored, reader,
                                      VkComponentMapping{});
        texture->normalEncoding = channels == 2 ? 3.0f : 2.0f;
    } else {
        const int channel = std::clamp(reader.channel, 0, 3);
        if (reader.scale[channel] != 1.0f ||
            reader.bias[channel] != 0.0f) {
            throw std::invalid_argument(
                "A stored one-channel texture is read without scale "
                "or "
                "bias");
        }
        // The reader's channel into r, which the shaders read; a
        // channel the format lacks reads as its alpha would (1) or,
        // for grey images, as the grey.
        constexpr std::array<VkComponentSwizzle, 4> kSwizzles{
            VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
            VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
        VkComponentSwizzle source =
            kSwizzles[static_cast<std::size_t>(channel)];
        if (static_cast<std::uint32_t>(channel) >= channels) {
            source = channel == 3 ? VK_COMPONENT_SWIZZLE_ONE
                                  : VK_COMPONENT_SWIZZLE_R;
        }
        texture->image = UploadStored(
            context, stored, reader,
            {source, source, source, VK_COMPONENT_SWIZZLE_ONE});
    }
    return texture;
}

RasterMaterial::RasterMaterial(const VulkanContext& context,
                               FrameUploads& uploads)
    : context_(context), uploads_(uploads),
      uniform_(context, sizeof(SurfaceUniforms),
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
               BufferMemory::Device),
      sampler_(context, SamplerDesc{VK_FILTER_LINEAR,
                                    VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                    16.0f, false}) {
    ClearAlbedoMap();
    SetDetailTextures(nullptr, nullptr);
    for (std::size_t slot = 0; slot < kMaterialTextureSlots; ++slot) {
        fallbacks_[slot] = FallbackTexture(
            context, static_cast<MaterialTextureSlot>(slot));
        samplers_[slot] = TextureSampler(context, TextureReader());
    }
    // With the fallbacks' heap slots.
    WriteUniform();
}

void RasterMaterial::SetTexture(
    MaterialTextureSlot slot,
    std::shared_ptr<const MaterialTexture> texture,
    const TextureReader& reader) {
    auto sampler = TextureSampler(context_, reader);
    vkDeviceWaitIdle(context_.Device());
    const auto index = static_cast<std::size_t>(slot);
    textures_[index] = std::move(texture);
    samplers_[index] = std::move(sampler);
    WriteUniform();
}

void RasterMaterial::AppendResources(
    std::vector<DescriptorResource>& resources) const {
    resources.push_back(DescriptorResource::Of(uniform_));
}

void RasterMaterial::SetAlbedoMap(const RasterView& albedo) {
    CheckRasterView(albedo, "Albedo map", 1);
    const std::uint32_t maxSize =
        context_.Limits().maxImageDimension2D;
    if (albedo.width > maxSize || albedo.height > maxSize) {
        throw std::invalid_argument(
            "Albedo map exceeds the device's maximum texture size");
    }
    const auto w = static_cast<std::uint32_t>(albedo.width);
    const auto h = static_cast<std::uint32_t>(albedo.height);
    std::vector<std::uint32_t> packed(static_cast<std::size_t>(w) *
                                      h);
    for (std::uint32_t row = 0; row < h; ++row) {
        for (std::uint32_t col = 0; col < w; ++col) {
            const float value = albedo(col, row);
            packed[static_cast<std::size_t>(row) * w + col] =
                glm::packHalf2x16(albedo.IsValid(value)
                                      ? glm::vec2(value, 1.0f)
                                      : glm::vec2(0.0f));
        }
    }
    auto image = std::make_unique<GpuImage>(
        context_, ImageDesc{w, h, 1, 1, VK_FORMAT_R16G16_SFLOAT,
                            VK_IMAGE_USAGE_SAMPLED_BIT});
    const std::span<const std::byte> mips[] = {Bytes(packed)};
    image->Upload(mips);
    vkDeviceWaitIdle(context_.Device());
    albedoMap_ = std::move(image);
    hasAlbedoMap_ = true;
    albedoGrid_ = glm::dvec4(albedo.originX, albedo.originY,
                             albedo.cellSize, 0.0);
    albedoSize_ = {w, h};
    WriteUniform();
}

void RasterMaterial::ClearAlbedoMap() {
    const std::uint32_t none = glm::packHalf2x16(glm::vec2(0.0f));
    auto image = std::make_unique<GpuImage>(
        context_, ImageDesc{1, 1, 1, 1, VK_FORMAT_R16G16_SFLOAT,
                            VK_IMAGE_USAGE_SAMPLED_BIT});
    const std::span<const std::byte> mips[] = {
        std::as_bytes(std::span(&none, 1))};
    image->Upload(mips);
    vkDeviceWaitIdle(context_.Device());
    albedoMap_ = std::move(image);
    hasAlbedoMap_ = false;
    albedoGrid_ = glm::dvec4(0.0);
    albedoSize_ = {1u, 1u};
    WriteUniform();
}

void RasterMaterial::SetDetailTextures(const Image* albedo,
                                       const Image* normal) {
    CheckImage(context_, albedo);
    CheckImage(context_, normal);
    SetDetailAlbedo(albedo);
    SetDetailNormal(normal);
}

void RasterMaterial::SetDetailAlbedo(const Image* albedo) {
    CheckImage(context_, albedo);
    const Image white(1, 1, glm::vec4(1.0f));
    auto image =
        UploadMips(context_, albedo != nullptr ? *albedo : white,
                   MipFilter::Average, ImageEncoding::Srgb,
                   VK_FORMAT_R8G8B8A8_SRGB);
    vkDeviceWaitIdle(context_.Device());
    detailAlbedo_ = std::move(image);
    hasDetailAlbedo_ = albedo != nullptr;
    detailMean_ = albedo != nullptr ? QuantisedSrgbMean(*albedo)
                                    : glm::vec3(1.0f);
    WriteUniform();
}

void RasterMaterial::SetDetailNormal(const Image* normal) {
    CheckImage(context_, normal);
    const Image flat(1, 1, glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
    auto image =
        UploadMips(context_, normal != nullptr ? *normal : flat,
                   MipFilter::NormalSlopes, ImageEncoding::Linear,
                   VK_FORMAT_R8G8B8A8_UNORM);
    vkDeviceWaitIdle(context_.Device());
    detailNormal_ = std::move(image);
    hasDetailNormal_ = normal != nullptr;
    WriteUniform();
}

void RasterMaterial::Update(const MaterialSettings& settings,
                            std::span<const float> brdfParams,
                            std::optional<glm::vec2> gloss) {
    gloss_ = gloss;
    const bool sameSpectrum =
        settings.albedo == settings_.albedo &&
        settings.reflectance.Wavelengths() ==
            settings_.reflectance.Wavelengths() &&
        settings.reflectance.Values() ==
            settings_.reflectance.Values();
    if (!sameSpectrum || reflectance_.Empty()) {
        reflectance_ =
            settings.reflectance.Empty()
                ? RgbToSpectrum(glm::dvec3(settings.albedo))
                : settings.reflectance;
        const glm::dvec3 rgb = ReflectanceToLinearSrgb(reflectance_);
        luminance_ = static_cast<float>(
            0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b);
    }
    settings_ = settings;
    params_.assign(brdfParams.begin(), brdfParams.end());
    WriteUniform();
}

void RasterMaterial::SetOrigin(const glm::dvec3& origin) {
    if (origin != origin_) {
        origin_ = origin;
        WriteUniform();
    }
}

void RasterMaterial::WriteUniform() {
    const MaterialSettings& settings = settings_;
    const std::span<const float> brdfParams = params_;
    SurfaceUniforms u{};
    u.albedo[0] = settings.albedo.r;
    u.albedo[1] = settings.albedo.g;
    u.albedo[2] = settings.albedo.b;
    u.albedo[3] = luminance_;
    const bool useMap = hasAlbedoMap_ && settings.useAlbedoMap;
    // Relative to the frame's origin, as the positions shading
    // samples it at are.
    u.albedoMap[0] = static_cast<float>(albedoGrid_.x - origin_.x);
    u.albedoMap[1] = static_cast<float>(albedoGrid_.y - origin_.y);
    u.albedoMap[2] = static_cast<float>(albedoGrid_.z);
    u.albedoMap[3] = useMap ? 1.0f : 0.0f;
    const DetailSettings& detail = settings.detail;
    u.albedoMapSize[0] = static_cast<float>(albedoSize_.x);
    u.albedoMapSize[1] = static_cast<float>(albedoSize_.y);
    u.albedoMapSize[2] = 1.0f / std::max(detail.tileSize, 1e-6f);
    u.albedoMapSize[3] = detail.directXNormals ? -1.0f : 1.0f;
    u.detail[0] = detail.albedoStrength;
    u.detail[1] = detail.normalStrength;
    u.detail[2] = hasDetailAlbedo_ ? 1.0f : 0.0f;
    u.detail[3] = hasDetailNormal_ ? 1.0f : 0.0f;
    u.detailMean[0] = detailMean_.r;
    u.detailMean[1] = detailMean_.g;
    u.detailMean[2] = detailMean_.b;
    u.hex[0] = detail.hexTiling ? 1.0f : 0.0f;
    u.hex[1] = std::clamp(detail.hexRotation, 0.0f, 1.0f);
    u.hex[2] = std::clamp(detail.hexContrast, 0.0f, 1.0f);
    u.hex[3] = std::max(detail.hexSharpness, 1.0f);
    // The detail tiling at the origin, so it stays fixed to the
    // world.
    const auto tilesPerUnit = static_cast<double>(u.albedoMapSize[2]);
    const glm::dvec3 tiles = origin_ * tilesPerUnit;
    const glm::dvec3 fraction = tiles - glm::floor(tiles);
    const glm::dvec2 xy = HexPhase({tiles.x, tiles.y});
    const glm::dvec2 yz = HexPhase({tiles.y, tiles.z});
    const glm::dvec2 xz = HexPhase({tiles.x, tiles.z});
    u.detailPhase[0] = static_cast<float>(fraction.x);
    u.detailPhase[1] = static_cast<float>(fraction.y);
    u.detailPhase[2] = static_cast<float>(fraction.z);
    u.hexPhaseXyYz[0] = static_cast<float>(xy.x);
    u.hexPhaseXyYz[1] = static_cast<float>(xy.y);
    u.hexPhaseXyYz[2] = static_cast<float>(yz.x);
    u.hexPhaseXyYz[3] = static_cast<float>(yz.y);
    u.hexPhaseXz[0] = static_cast<float>(xz.x);
    u.hexPhaseXz[1] = static_cast<float>(xz.y);
    const TextureReader albedoReader =
        settings.albedoTexture.value_or(TextureReader());
    const TextureReader normalReader =
        settings.normalTexture.value_or(TextureReader());
    for (int c = 0; c < 4; ++c) {
        u.albedoTextureScale[c] = albedoReader.scale[c];
        u.albedoTextureBias[c] = albedoReader.bias[c];
    }
    WriteUvTransform(albedoReader, u.albedoUv);
    WriteUvTransform(normalReader, u.normalUv);
    WriteUvTransform(
        settings.roughnessTexture.value_or(TextureReader()),
        u.roughnessUv);
    WriteUvTransform(
        settings.metallicTexture.value_or(TextureReader()),
        u.metallicUv);
    WriteUvTransform(
        settings.opacityTexture.value_or(TextureReader()),
        u.opacityUv);
    using Slot = MaterialTextureSlot;
    const auto bound = [&](Slot slot, bool wanted) {
        return wanted && HasTexture(slot) ? 1.0f : 0.0f;
    };
    u.textures[0] =
        bound(Slot::Albedo, settings.albedoTexture.has_value());
    // A normal map's flag is its encoding (`MaterialTexture`).
    const MaterialTexture* normal =
        textures_[static_cast<std::size_t>(Slot::Normal)].get();
    u.textures[1] = settings.normalTexture && normal != nullptr
                        ? normal->normalEncoding
                        : 0.0f;
    u.textures[2] =
        bound(Slot::Roughness, settings.roughnessTexture.has_value());
    u.textures[3] =
        bound(Slot::Metallic, settings.metallicTexture.has_value());
    u.opacity[0] = settings.opacity;
    u.opacity[1] = settings.opacityThreshold;
    u.opacity[2] =
        bound(Slot::Opacity, settings.opacityTexture.has_value());
    const std::size_t count =
        std::min(brdfParams.size(), BrdfRegistry::kMaxParams);
    if (count > 0) {
        std::memcpy(u.params, brdfParams.data(),
                    count * sizeof(float));
    }
    if (gloss_) {
        u.gloss[0] = gloss_->x;
        u.gloss[1] = gloss_->y;
        u.gloss[2] = 1.0f;
    }
    u.emission[0] = settings.emission.r;
    u.emission[1] = settings.emission.g;
    u.emission[2] = settings.emission.b;
    // The textures and samplers in the bindless heap, in fresh slots
    // with each write: the last ones retire once no frame in flight
    // reads them. Slot 0 stands for one not made yet, never sampled.
    BindlessHeap& heap = context_.Bindless();
    heapSlots_.clear();
    const auto image = [&](const GpuImage* texture) -> std::uint32_t {
        if (texture == nullptr) {
            return 0;
        }
        heapSlots_.push_back(heap.AddImage(
            BindlessHeap::Kind::Texture2D, texture->View()));
        return heapSlots_.back().Index();
    };
    const auto sampler = [&](const Sampler* filter) -> std::uint32_t {
        if (filter == nullptr) {
            return 0;
        }
        heapSlots_.push_back(heap.AddSampler(filter->Handle()));
        return heapSlots_.back().Index();
    };
    u.heap[0] = image(albedoMap_.get());
    u.heap[1] = image(detailAlbedo_.get());
    u.heap[2] = image(detailNormal_.get());
    u.heap[3] = sampler(&sampler_);
    for (std::size_t slot = 0; slot < kMaterialTextureSlots; ++slot) {
        u.heap[4 + slot] = image(textures_[slot] != nullptr
                                     ? textures_[slot]->image.get()
                                     : fallbacks_[slot].get());
        u.heap[9 + slot] = sampler(samplers_[slot].get());
    }
    // A virtual albedo texture (`textures.x` 2 on a sparse image, 3
    // in pages): the sparse image and its min-LOD map (slot 14), or
    // the tail image and the pages (slot 14, an array); this frame
    // slot's request flags, which the page table follows.
    const MaterialTexture* albedoTexture = textures_[0].get();
    if (albedoTexture != nullptr && albedoTexture->virtualTexture &&
        u.textures[0] > 0.0f) {
        const VirtualTexture& texture =
            *albedoTexture->virtualTexture;
        const TileStore& store = texture.Store();
        if (const SparseImage* sparse = texture.Sparse()) {
            heapSlots_.push_back(heap.AddImage(
                BindlessHeap::Kind::Texture2D, sparse->View()));
            u.heap[4] = heapSlots_.back().Index();
            u.heap[14] = image(&texture.MinLodMap(frameSlot_));
            u.textures[0] = 2.0f;
        } else {
            const PagePool& pool = *texture.Pool();
            u.heap[4] = image(&pool.TailImage());
            heapSlots_.push_back(
                heap.AddImage(BindlessHeap::Kind::Texture2DArray,
                              pool.PageImage().View()));
            u.heap[14] = heapSlots_.back().Index();
            u.textures[0] = 3.0f;
        }
        u.virtualAlbedo[0] = store.TailMip();
        u.virtualAlbedo[1] = texture.Mips();
        u.virtualAlbedo[2] = texture.Width();
        u.virtualAlbedo[3] = texture.Height();
        u.virtualTile[0] = store.TileTexels().width;
        u.virtualTile[1] = store.TileTexels().height;
        u.virtualRequests = texture.RequestAddress(frameSlot_);
    }
    uploads_.Write(uniform_, std::as_bytes(std::span(&u, 1)));
}

void RasterMaterial::BeginFrame(std::uint32_t slot,
                                std::uint64_t frame) {
    const MaterialTexture* albedoTexture = textures_[0].get();
    if (albedoTexture == nullptr || !albedoTexture->virtualTexture) {
        return;
    }
    albedoTexture->virtualTexture->BeginFrame(slot, frame);
    frameSlot_ = slot;
    WriteUniform();
}

std::uint64_t RasterMaterial::GpuBytes() const {
    std::uint64_t bytes = uniform_.Size() + albedoMap_->Bytes() +
                          detailAlbedo_->Bytes() +
                          detailNormal_->Bytes();
    for (const auto& fallback : fallbacks_) {
        bytes += fallback->Bytes();
    }
    return bytes;
}

}  // namespace ernie

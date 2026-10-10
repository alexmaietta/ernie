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
    float albedoUv[8];  ///< Rows of the 2 x 3 texture coordinate transform.
    float normalUv[8];
    float roughnessUv[8];
    float metallicUv[8];
    float textures[4];  ///< Has albedo texture, normal map, roughness, metallic.
    float opacityUv[8];
    float opacity[4];  ///< Opacity, its cutout threshold, has a texture, unused.
    float params[BrdfRegistry::kMaxParams];
};
static_assert(sizeof(SurfaceUniforms) == 432,
              "SurfaceUniforms must match Surface in surface.slang");

/// UsdTransform2d as the rows of a 2 x 3 matrix: scale, then rotate counter-clockwise, then
/// translate.
void WriteUvTransform(const TextureReader& reader, float (&rows)[8]) {
    const float angle = glm::radians(reader.uvRotation);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const glm::vec2 k = reader.uvScale;
    const float values[8] = {c * k.x, -s * k.y, reader.uvTranslation.x, 0.0f,
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

/// A trilinear, anisotropic sampler wrapping as `reader` does.
std::unique_ptr<Sampler> TextureSampler(const VulkanContext& context, const TextureReader& reader) {
    SamplerDesc desc{VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, AddressModeOf(reader.wrapS),
                     16.0f, false};
    desc.addressModeV = AddressModeOf(reader.wrapT);
    desc.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    return std::make_unique<Sampler>(context, desc);
}

/// surface.slang's hex lattice: its skew (tile coordinates times `kSkewScale`, then x minus
/// `kSkewX` times y and y times `kSkewY`) and the period of its random rotations.
constexpr double kSkewScale = 3.4641016151377544;  // 2 sqrt(3)
constexpr double kSkewX = 0.57735027;
constexpr double kSkewY = 1.15470054;
constexpr double kHexPeriod = 4096.0;

/// The hex lattice's phase at world plane point `planar` (in tiles): the tile offset that puts the
/// origin-relative lattice where the world's is, modulo the rotations' period.
glm::dvec2 HexPhase(const glm::dvec2& planar) {
    const glm::dvec2 st = planar * kSkewScale;
    glm::dvec2 skewed(st.x - kSkewX * st.y, kSkewY * st.y);
    skewed -= kHexPeriod * glm::floor(skewed / kHexPeriod);
    const double y = skewed.y / (kSkewScale * kSkewY);
    return {skewed.x / kSkewScale + kSkewX * y, y};
}

glm::vec3 QuantisedSrgbMean(const Image& image) {
    const std::vector<std::uint8_t> bytes = ToRgba8(image, ImageEncoding::Srgb);
    glm::dvec3 sum(0.0);
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        for (int c = 0; c < 3; ++c) {
            const float stored = static_cast<float>(bytes[4 * i + static_cast<std::size_t>(c)]);
            sum[c] += static_cast<double>(SrgbToLinear(stored / 255.0f));
        }
    }
    return glm::vec3(sum / static_cast<double>(image.pixels.size()));
}

std::span<const std::byte> Bytes(const auto& values) {
    return std::as_bytes(std::span(values));
}

}  // namespace

RasterMaterial::RasterMaterial(const VulkanContext& context, FrameUploads& uploads)
    : context_(context), uploads_(uploads),
      uniform_(context, sizeof(SurfaceUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
               BufferMemory::Device),
      sampler_(context, SamplerDesc{VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                    VK_SAMPLER_ADDRESS_MODE_REPEAT, 16.0f, false}) {
    ClearAlbedoMap();
    SetDetailTextures(nullptr, nullptr);
    SetAlbedoTexture(nullptr, TextureReader());
    SetNormalTexture(nullptr, TextureReader());
    for (const Scalar which : {Scalar::Roughness, Scalar::Metallic, Scalar::Opacity}) {
        SetScalarTexture(which, nullptr, TextureReader());
    }
}

void RasterMaterial::SetScalarTexture(Scalar which, const Image* image,
                                      const TextureReader& reader) {
    CheckDetailImage(image);
    Image values(1, 1, glm::vec4(0.0f));
    if (image != nullptr) {
        values = *image;
        const int channel = std::clamp(reader.channel, 0, 3);
        for (glm::vec4& texel : values.pixels) {
            const float value = std::clamp(
                texel[channel] * reader.scale[channel] + reader.bias[channel], 0.0f, 1.0f);
            texel = glm::vec4(value, value, value, 1.0f);
        }
    }
    auto texture =
        UploadMips(values, MipFilter::Average, ImageEncoding::Linear, VK_FORMAT_R8G8B8A8_UNORM);
    auto sampler = TextureSampler(context_, reader);
    vkDeviceWaitIdle(context_.Device());
    const auto slot = static_cast<std::size_t>(which);
    scalarTextures_[slot] = std::move(texture);
    scalarSamplers_[slot] = std::move(sampler);
    hasScalarTexture_[slot] = image != nullptr;
    WriteUniform();
}

void RasterMaterial::SetAlbedoTexture(const Image* albedo, const TextureReader& reader) {
    CheckDetailImage(albedo);
    const Image white(1, 1, glm::vec4(1.0f));
    auto image = UploadMips(albedo != nullptr ? *albedo : white, MipFilter::Average,
                            ImageEncoding::Srgb, VK_FORMAT_R8G8B8A8_SRGB);
    auto sampler = TextureSampler(context_, reader);
    vkDeviceWaitIdle(context_.Device());
    albedoTexture_ = std::move(image);
    albedoSampler_ = std::move(sampler);
    hasAlbedoTexture_ = albedo != nullptr;
    WriteUniform();
}

void RasterMaterial::SetNormalTexture(const Image* normal, const TextureReader& reader) {
    CheckDetailImage(normal);
    // The reader's scale and bias turn the raw values into normals, stored as n / 2 + 1/2 for
    // `MipFilter::NormalSlopes`.
    Image normals(1, 1, glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
    if (normal != nullptr) {
        normals = *normal;
        for (glm::vec4& texel : normals.pixels) {
            glm::vec3 n = glm::vec3(texel) * glm::vec3(reader.scale) + glm::vec3(reader.bias);
            n = glm::length(n) > 0.0f ? glm::normalize(n) : glm::vec3(0.0f, 0.0f, 1.0f);
            texel = glm::vec4(0.5f * n + 0.5f, 0.0f);
        }
    }
    auto image = UploadMips(normals, MipFilter::NormalSlopes, ImageEncoding::Linear,
                            VK_FORMAT_R8G8B8A8_UNORM);
    auto sampler = TextureSampler(context_, reader);
    vkDeviceWaitIdle(context_.Device());
    normalTexture_ = std::move(image);
    normalSampler_ = std::move(sampler);
    hasNormalTexture_ = normal != nullptr;
    WriteUniform();
}

void RasterMaterial::AppendTextureResources(std::vector<DescriptorResource>& resources) const {
    resources.push_back(DescriptorResource::Of(
        albedoTexture_->View(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, *albedoSampler_));
    resources.push_back(DescriptorResource::Of(
        normalTexture_->View(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, *normalSampler_));
    for (std::size_t slot = 0; slot < scalarTextures_.size(); ++slot) {
        resources.push_back(DescriptorResource::Of(scalarTextures_[slot]->View(),
                                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                   *scalarSamplers_[slot]));
    }
}

void RasterMaterial::AppendResources(std::vector<DescriptorResource>& resources) const {
    resources.push_back(DescriptorResource::Of(uniform_));
    resources.push_back(DescriptorResource::Of(albedoMap_->View()));
    resources.push_back(DescriptorResource::Of(detailAlbedo_->View()));
    resources.push_back(DescriptorResource::Of(detailNormal_->View()));
    resources.push_back(DescriptorResource::Of(sampler_));
}

void RasterMaterial::SetAlbedoMap(const RasterView& albedo) {
    CheckRasterView(albedo, "Albedo map", 1);
    const std::uint32_t maxSize = context_.Limits().maxImageDimension2D;
    if (albedo.width > maxSize || albedo.height > maxSize) {
        throw std::invalid_argument("Albedo map exceeds the device's maximum texture size");
    }
    const auto w = static_cast<std::uint32_t>(albedo.width);
    const auto h = static_cast<std::uint32_t>(albedo.height);
    std::vector<std::uint32_t> packed(static_cast<std::size_t>(w) * h);
    for (std::uint32_t row = 0; row < h; ++row) {
        for (std::uint32_t col = 0; col < w; ++col) {
            const float value = albedo(col, row);
            packed[static_cast<std::size_t>(row) * w + col] =
                glm::packHalf2x16(albedo.IsValid(value) ? glm::vec2(value, 1.0f) : glm::vec2(0.0f));
        }
    }
    auto image = std::make_unique<GpuImage>(
        context_, ImageDesc{w, h, 1, 1, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT});
    const std::span<const std::byte> mips[] = {Bytes(packed)};
    image->Upload(mips);
    vkDeviceWaitIdle(context_.Device());
    albedoMap_ = std::move(image);
    hasAlbedoMap_ = true;
    albedoGrid_ = glm::dvec4(albedo.originX, albedo.originY, albedo.cellSize, 0.0);
    albedoSize_ = {w, h};
    WriteUniform();
}

void RasterMaterial::ClearAlbedoMap() {
    const std::uint32_t none = glm::packHalf2x16(glm::vec2(0.0f));
    auto image = std::make_unique<GpuImage>(
        context_, ImageDesc{1, 1, 1, 1, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT});
    const std::span<const std::byte> mips[] = {std::as_bytes(std::span(&none, 1))};
    image->Upload(mips);
    vkDeviceWaitIdle(context_.Device());
    albedoMap_ = std::move(image);
    hasAlbedoMap_ = false;
    albedoGrid_ = glm::dvec4(0.0);
    albedoSize_ = {1u, 1u};
    WriteUniform();
}

void RasterMaterial::SetDetailTextures(const Image* albedo, const Image* normal) {
    CheckDetailImage(albedo);
    CheckDetailImage(normal);
    SetDetailAlbedo(albedo);
    SetDetailNormal(normal);
}

void RasterMaterial::SetDetailAlbedo(const Image* albedo) {
    CheckDetailImage(albedo);
    const Image white(1, 1, glm::vec4(1.0f));
    auto image = UploadMips(albedo != nullptr ? *albedo : white, MipFilter::Average,
                            ImageEncoding::Srgb, VK_FORMAT_R8G8B8A8_SRGB);
    vkDeviceWaitIdle(context_.Device());
    detailAlbedo_ = std::move(image);
    hasDetailAlbedo_ = albedo != nullptr;
    detailMean_ = albedo != nullptr ? QuantisedSrgbMean(*albedo) : glm::vec3(1.0f);
    WriteUniform();
}

void RasterMaterial::SetDetailNormal(const Image* normal) {
    CheckDetailImage(normal);
    const Image flat(1, 1, glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
    auto image = UploadMips(normal != nullptr ? *normal : flat, MipFilter::NormalSlopes,
                            ImageEncoding::Linear, VK_FORMAT_R8G8B8A8_UNORM);
    vkDeviceWaitIdle(context_.Device());
    detailNormal_ = std::move(image);
    hasDetailNormal_ = normal != nullptr;
    WriteUniform();
}

void RasterMaterial::CheckDetailImage(const Image* image) const {
    if (image == nullptr) {
        return;
    }
    if (image->width == 0 || image->height == 0 ||
        image->pixels.size() != static_cast<std::size_t>(image->width) * image->height) {
        throw std::invalid_argument("Detail texture is empty");
    }
    const std::uint32_t maxSize = context_.Limits().maxImageDimension2D;
    if (image->width > maxSize || image->height > maxSize) {
        throw std::invalid_argument("Detail texture exceeds the device's maximum texture size");
    }
}

void RasterMaterial::Update(const MaterialSettings& settings, std::span<const float> brdfParams) {
    const bool sameSpectrum =
        settings.albedo == settings_.albedo &&
        settings.reflectance.Wavelengths() == settings_.reflectance.Wavelengths() &&
        settings.reflectance.Values() == settings_.reflectance.Values();
    if (!sameSpectrum || reflectance_.Empty()) {
        reflectance_ = settings.reflectance.Empty() ? RgbToSpectrum(glm::dvec3(settings.albedo))
                                                    : settings.reflectance;
        const glm::dvec3 rgb = ReflectanceToLinearSrgb(reflectance_);
        luminance_ = static_cast<float>(0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b);
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
    // Relative to the frame's origin, as the positions shading samples it at are.
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
    // The detail tiling at the origin, so it stays fixed to the world.
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
    const TextureReader albedoReader = settings.albedoTexture.value_or(TextureReader());
    const TextureReader normalReader = settings.normalTexture.value_or(TextureReader());
    for (int c = 0; c < 4; ++c) {
        u.albedoTextureScale[c] = albedoReader.scale[c];
        u.albedoTextureBias[c] = albedoReader.bias[c];
    }
    WriteUvTransform(albedoReader, u.albedoUv);
    WriteUvTransform(normalReader, u.normalUv);
    WriteUvTransform(settings.roughnessTexture.value_or(TextureReader()), u.roughnessUv);
    WriteUvTransform(settings.metallicTexture.value_or(TextureReader()), u.metallicUv);
    WriteUvTransform(settings.opacityTexture.value_or(TextureReader()), u.opacityUv);
    u.textures[0] = hasAlbedoTexture_ && settings.albedoTexture ? 1.0f : 0.0f;
    u.textures[1] = hasNormalTexture_ && settings.normalTexture ? 1.0f : 0.0f;
    u.textures[2] = hasScalarTexture_[0] && settings.roughnessTexture ? 1.0f : 0.0f;
    u.textures[3] = hasScalarTexture_[1] && settings.metallicTexture ? 1.0f : 0.0f;
    u.opacity[0] = settings.opacity;
    u.opacity[1] = settings.opacityThreshold;
    u.opacity[2] = hasScalarTexture_[2] && settings.opacityTexture ? 1.0f : 0.0f;
    const std::size_t count = std::min(brdfParams.size(), BrdfRegistry::kMaxParams);
    std::memcpy(u.params, brdfParams.data(), count * sizeof(float));
    uploads_.Write(uniform_, std::as_bytes(std::span(&u, 1)));
}

std::uint64_t RasterMaterial::GpuBytes() const {
    return uniform_.Size() + albedoMap_->Bytes() + detailAlbedo_->Bytes() + detailNormal_->Bytes() +
           albedoTexture_->Bytes() + normalTexture_->Bytes() + scalarTextures_[0]->Bytes() +
           scalarTextures_[1]->Bytes() + scalarTextures_[2]->Bytes();
}

std::unique_ptr<GpuImage> RasterMaterial::UploadMips(const Image& image, MipFilter filter,
                                                     ImageEncoding encoding,
                                                     VkFormat format) const {
    std::vector<Image> mips = MipChain(image, filter);
    if (filter == MipFilter::NormalSlopes) {
        // Normals stored as n * 127 + 128 in 8 bits, so flat is exactly byte 128 and decodes to
        // exactly 0 (surface.slang's `detailTap`); n / 2 + 1/2 would store 0 as 127.5, tilting
        // every normal by 1/255. Source maps that stored flat as 128 decode to flat too.
        for (Image& mip : mips) {
            for (glm::vec4& texel : mip.pixels) {
                const glm::vec3 n = 2.0f * glm::vec3(texel) - 1.0f;
                const glm::vec3 bytes = glm::round(glm::clamp(n * 127.0f + 128.0f, 1.0f, 255.0f));
                texel = glm::vec4(bytes / 255.0f, texel.a);
            }
        }
    }
    auto texture = std::make_unique<GpuImage>(
        context_, ImageDesc{image.width, image.height, 1, static_cast<std::uint32_t>(mips.size()),
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

}  // namespace ernie

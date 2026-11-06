// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string>

#include "ernie/device/kernels/texture.h"
#include "ernie/device/resource.h"
#include "ernie/export.h"

namespace ernie::device {

struct TextureDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t levels = 1;
    TexelFormat format = TexelFormat::Rgba8;
    TexelColorSpace colorSpace = TexelColorSpace::Linear;
    TextureWrap wrapS = TextureWrap::Repeat;
    TextureWrap wrapT = TextureWrap::Repeat;
    std::string label = "texture";
};

class ERNIE_API Texture : public Resource {
  public:
    ~Texture() override;

    const TextureDesc& Desc() const {
        return m_desc;
    }

    uint32_t Handle() const {
        return m_handle;
    }

    virtual uint64_t KernelHandle(TextureWrap wrapS, TextureWrap wrapT) = 0;

    uint64_t LevelBytes(uint32_t level) const {
        return TextureLevelBytes(m_desc.format, m_desc.width, m_desc.height, level);
    }

  protected:
    Texture(Device& device, const TextureDesc& desc) : Resource(device, desc.label), m_desc(desc) {}

    uint32_t m_handle = 0;

  private:
    friend class Device;

    TextureDesc m_desc;
    bool m_counted = false;
};

}

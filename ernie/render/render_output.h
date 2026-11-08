// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/device/operation.h"
#include "ernie/device/tensor.h"
#include "ernie/export.h"
#include "ernie/io/image.h"
#include "ernie/scene/types.h"

namespace ernie {

struct ERNIE_API RenderVarOutput {
    std::string renderVarPath;
    std::string sourceName;
    device::Tensor tensor;
    std::shared_ptr<const IdMap> idMap;
    std::map<std::string, device::Tensor, std::less<>> channels;
};

struct ERNIE_API FrameOutput {
    double startTime = 0;
    double endTime = 0;
    std::map<std::string, RenderVarOutput, std::less<>> renderVars;
    device::Tensor overlay;

    const RenderVarOutput& RenderVar(std::string_view path) const;
    const RenderVarOutput& Source(std::string_view sourceName) const;
};

struct ERNIE_API ProductOutput {
    std::string name;
    std::vector<FrameOutput> frames;
};

ERNIE_API IdTexts DecodeIdMap(const device::Tensor& map);

ERNIE_API std::filesystem::path
WriteRenderVar(std::filesystem::path path, const RenderVarOutput& renderVar, int64_t camera = 0);
ERNIE_API std::vector<std::filesystem::path> WriteFrame(const std::filesystem::path& directory,
                                                        const FrameOutput& frame, bool raw = false);

class ERNIE_API ImageWriter {
  public:
    explicit ImageWriter(uint32_t threads = 4, uint32_t maxPending = 64);
    ~ImageWriter();
    ImageWriter(const ImageWriter&) = delete;
    ImageWriter& operator=(const ImageWriter&) = delete;

    void WriteImage(std::filesystem::path path, Pixels image);
    void WriteImage(std::filesystem::path path, const device::Tensor& image);
    void WriteRenderVar(std::filesystem::path path, const RenderVarOutput& renderVar,
                        int64_t camera = 0);
    std::vector<std::filesystem::path> WriteFrame(const std::filesystem::path& directory,
                                                  const FrameOutput& frame, bool raw = false);
    void Wait();
    size_t Pending() const;

  private:
    struct State;
    std::unique_ptr<State> m_state;
};

class ERNIE_API RenderOutput {
  public:
    double simulationStartTime = 0;
    double simulationEndTime = 0;

    bool Ready() const {
        return m_done.Ready();
    }

    void Wait() const {
        m_done.Wait();
    }

    const device::Operation& Done() const {
        return m_done;
    }

    const ProductOutput& operator[](std::string_view product) const;

    bool Contains(std::string_view product) const {
        return m_products.find(product) != m_products.end();
    }

    size_t Size() const {
        return m_products.size();
    }

    auto begin() const {
        return m_products.begin();
    }

    auto end() const {
        return m_products.end();
    }

  private:
    friend class RendererImpl;

    device::Operation m_done;
    std::map<std::string, ProductOutput, std::less<>> m_products;
};

}

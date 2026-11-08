// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/device/device.h"
#include "ernie/device/tensor.h"
#include "ernie/export.h"
#include "ernie/scene/scene.h"

namespace ernie {

namespace pipeline {
struct PipelineDesc;
class NodeRegistry;
}

struct PipelineNodePort {
    std::string name;
    std::string type;
    bool optional = false;
};

struct PipelineNodeInfo {
    std::string id;
    std::string kind;
    std::vector<PipelineNodePort> inputs;
    std::vector<PipelineNodePort> outputs;
};

ERNIE_API std::optional<PipelineNodeInfo> FindPipelineNode(std::string_view id);

struct DetectorDesc {
    DetectorPattern pattern = DetectorPattern::Rgb;
    float fullWell = 10000;
    float readNoise = 2;
    float darkCurrent = 1;
    float darkCurrentDoubling = 7;
    float temperature = 293.15f;
    float prnu = 0.005f;
    float dsnu = 0.5f;
    float systemGain = 0.25f;
    uint32_t bitDepth = 12;
    float blackLevel = 0;
    uint32_t seed = 0;
};

struct PipelineRunDesc {
    std::map<std::string, device::Tensor, std::less<>> inputs;
    DetectorDesc detector;
    float exposureTime = 1;
    float iso = 100;
    uint32_t capture = 0;
    uint32_t identity = 0;
};

using PipelineOutputs = std::map<std::string, device::Tensor, std::less<>>;

class ERNIE_API Pipeline {
  public:
    Pipeline(const Ref<Scene>& scene, std::string_view path);
    Pipeline(device::Device& device, const pipeline::PipelineDesc& desc);
    Pipeline(device::Device& device, const pipeline::PipelineDesc& desc,
             const pipeline::NodeRegistry& registry);
    ~Pipeline();
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;

    PipelineOutputs Run(const PipelineRunDesc& desc) const;

    device::Device& GetDevice() const;
    std::vector<std::string> Sources() const;
    std::vector<std::string> Outputs() const;

  private:
    friend class PipelineSession;
    struct State;

    PipelineOutputs Run(const PipelineRunDesc& desc, device::Tensor& state) const;

    std::unique_ptr<State> m_state;
};

class ERNIE_API PipelineSession {
  public:
    explicit PipelineSession(const Pipeline& pipeline) : m_pipeline(pipeline) {}

    PipelineOutputs Run(PipelineRunDesc desc);

    uint32_t Captures() const {
        return m_captures;
    }

    const Pipeline& GetPipeline() const {
        return m_pipeline;
    }

  private:
    const Pipeline& m_pipeline;
    uint32_t m_captures = 0;
    device::Tensor m_state;
};

}

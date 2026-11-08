// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "ernie/device/device.h"
#include "ernie/device/tensor.h"
#include "ernie/pipeline/resolve.h"
#include "ernie/scene/scene.h"

namespace ernie {

struct CompiledPipeline;

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

class Pipeline {
  public:
    Pipeline(device::Device& device, pipeline::PipelineDesc desc);
    Pipeline(device::Device& device, pipeline::PipelineDesc desc,
             const pipeline::NodeRegistry& registry);

    PipelineOutputs Run(const PipelineRunDesc& desc) const;

  private:
    friend class PipelineSession;

    PipelineOutputs Run(const PipelineRunDesc& desc, Ref<device::Buffer>& state) const;

    device::Device* m_device;
    const pipeline::NodeRegistry* m_registry;
    pipeline::ResolvedPipeline m_instance;
    std::shared_ptr<const CompiledPipeline> m_compiled;
};

class PipelineSession {
  public:
    explicit PipelineSession(const Pipeline& pipeline) : m_pipeline(pipeline) {}

    PipelineOutputs Run(PipelineRunDesc desc);

    uint32_t Captures() const {
        return m_captures;
    }

  private:
    const Pipeline& m_pipeline;
    uint32_t m_captures = 0;
    Ref<device::Buffer> m_state;
};

}

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

namespace sensor_pipeline {
struct SensorPipelineDesc;
class NodeRegistry;
}

struct SensorPipelineNodePort {
    std::string name;
    std::string type;
    bool optional = false;
};

struct SensorPipelineNodeInfo {
    std::string id;
    std::string kind;
    std::vector<SensorPipelineNodePort> inputs;
    std::vector<SensorPipelineNodePort> outputs;
};

ERNIE_API std::optional<SensorPipelineNodeInfo> FindSensorPipelineNode(std::string_view id);

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

struct SensorPipelineRunDesc {
    std::map<std::string, device::Tensor, std::less<>> inputs;
    DetectorDesc detector;
    float exposureTime = 1;
    float iso = 100;
    uint32_t capture = 0;
    uint32_t identity = 0;
};

using SensorPipelineOutputs = std::map<std::string, device::Tensor, std::less<>>;

class ERNIE_API SensorPipelineRunner {
  public:
    SensorPipelineRunner(const Ref<Scene>& scene, std::string_view path);
    SensorPipelineRunner(device::Device& device, const sensor_pipeline::SensorPipelineDesc& desc);
    SensorPipelineRunner(device::Device& device, const sensor_pipeline::SensorPipelineDesc& desc,
                         const sensor_pipeline::NodeRegistry& registry);
    ~SensorPipelineRunner();
    SensorPipelineRunner(SensorPipelineRunner&&) noexcept;
    SensorPipelineRunner& operator=(SensorPipelineRunner&&) noexcept;

    SensorPipelineOutputs Run(const SensorPipelineRunDesc& desc) const;

    device::Device& GetDevice() const;
    std::vector<std::string> Sources() const;
    std::vector<std::string> Outputs() const;

  private:
    friend class SensorPipelineSession;
    struct State;

    SensorPipelineOutputs Run(const SensorPipelineRunDesc& desc, device::Tensor& state) const;

    std::unique_ptr<State> m_state;
};

class ERNIE_API SensorPipelineSession {
  public:
    explicit SensorPipelineSession(const SensorPipelineRunner& pipeline) : m_pipeline(pipeline) {}

    SensorPipelineOutputs Run(SensorPipelineRunDesc desc);

    uint32_t Captures() const {
        return m_captures;
    }

    const SensorPipelineRunner& GetRunner() const {
        return m_pipeline;
    }

  private:
    const SensorPipelineRunner& m_pipeline;
    uint32_t m_captures = 0;
    device::Tensor m_state;
};

}

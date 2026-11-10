// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ernie::pipeline {

enum class SignalType : uint8_t {
    ElectronRate,
    Electrons,
    Digital,
    LinearColor,
    DisplayColor,
    Radiance,
    Returns,
    PointCloud,
    RadarCube,
    CaptureStat,
};

struct SignalInfo {
    std::string_view name;
    std::string_view slangType;
    std::string_view port;
};

inline constexpr SignalInfo kSignals[] = {
    {"ElectronRate", "ElectronRate", "electronRate"},
    {"Electrons", "Electrons", "electrons"},
    {"Digital", "Digital", "digital"},
    {"LinearColor", "LinearColor", "color"},
    {"DisplayColor", "DisplayColor", "display"},
    {"Radiance", "Radiance", "radiance"},
    {"Returns", "Returns", "returns"},
    {"PointCloud", "CloudSignal", "cloud"},
    {"RadarCube", "RadarCube", "cube"},
    {"CaptureStat", "CaptureStat", "stat"},
};

inline const SignalInfo& Signal(SignalType type) {
    return kSignals[uint32_t(type)];
}

inline constexpr uint32_t kReturnsChannels = 8;

enum class ElementType : uint8_t {
    F32,
    F16,
    U16,
    U8,
};

inline constexpr uint32_t kElementBytes[] = {4, 2, 2, 1};

inline constexpr uint32_t kStatWords = 16;
inline constexpr uint32_t kReduceGroups = 64;
inline constexpr uint32_t kStatStorageWords = (1 + kReduceGroups) * kStatWords;

struct SignalLayout {
    uint32_t channels = 1;
    uint32_t pattern = 0;
    bool mosaic = false;
    ElementType element = ElementType::F32;
    bool perCapture = false;

    bool operator==(const SignalLayout&) const = default;
};

enum class NodeKind : uint8_t {
    Source,
    Point,
    Stencil,
    Reduce,
    Global,
    Compact,
};

inline constexpr std::string_view kNodeKindNames[] = {"source", "point",  "stencil",
                                                      "reduce", "global", "compact"};

enum class LayoutRule : uint8_t {
    FirstInput,
    Detector,
    FullChannels,
    Color,
    Scalar,
    Stat,
    Returns,
};

inline constexpr uint32_t kMostStencilRadius = 8;

struct PortDef {
    std::string name;
    SignalType type = SignalType::ElectronRate;
    bool optional = false;
};

inline constexpr uint32_t kNoContextArgument = ~0u;

enum class ParameterType : uint8_t { Float, Int, Bool };

struct ParameterDef {
    std::string name;
    uint32_t words = 1;
    std::vector<float> fallback;
    ParameterType type = ParameterType::Float;
};

struct UniformDef {
    std::string name;
    std::vector<std::string> allowed;
    std::string fallback;
};

struct NodeCode {
    std::string module;
    std::string type;
    std::string file;
    std::string source;
    std::string version;
    bool applies = false;
    uint32_t context = kNoContextArgument;
};

struct NodeDef {
    std::string id;
    NodeKind kind = NodeKind::Point;
    std::vector<PortDef> inputs;
    std::vector<PortDef> outputs;
    std::vector<ParameterDef> parameters;
    std::vector<UniformDef> uniforms;
    std::vector<float> state;
    uint32_t radius = 0;
    uint32_t passes = 1;
    LayoutRule layout = LayoutRule::FirstInput;
    NodeCode code;
};

class NodeRegistry {
  public:
    uint32_t Add(NodeDef def);

    const NodeDef& operator[](uint32_t index) const {
        return m_defs[index];
    }

    uint32_t Find(std::string_view id) const;

    uint32_t Size() const {
        return uint32_t(m_defs.size());
    }

  private:
    std::deque<NodeDef> m_defs;
    std::unordered_map<std::string, uint32_t> m_ids;
};

inline constexpr uint32_t kNoDef = ~0u;

inline constexpr const char* kCameraNodes = "ernie.render.kernels.pipeline.nodes.camera";
inline constexpr const char* kIspNodes = "ernie.render.kernels.pipeline.nodes.isp";
inline constexpr const char* kLidarNodes = "ernie.render.kernels.pipeline.nodes.lidar";
inline constexpr const char* kRadarNodes = "ernie.render.kernels.pipeline.nodes.radar";
inline constexpr uint32_t kCfarRadius = 4;

void RegisterBuiltinNodes(NodeRegistry& registry);

const NodeRegistry& BuiltinNodes();

std::shared_ptr<const NodeRegistry> PipelineNodes();

void RegisterPipelineNode(NodeDef def);

}

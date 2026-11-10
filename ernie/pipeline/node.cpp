// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/pipeline/node.h"

#include <algorithm>

#include "ernie/graph/library.h"
#include "ernie/util/error.h"

namespace ernie::pipeline {

namespace {

void Validate(const NodeDef& def) {
    switch (def.kind) {
    case NodeKind::Source:
        if (!def.inputs.empty()) {
            throw Error("Pipeline node {} is a source, so it takes no signal inputs", def.id);
        }
        break;
    case NodeKind::Stencil:
        if (def.inputs.empty() || def.radius > kMostStencilRadius) {
            throw Error(
                "Pipeline node {} is a stencil, so it needs a signal input and a radius up to {}",
                def.id, kMostStencilRadius);
        }
        break;
    case NodeKind::Reduce:
        if (def.inputs.empty() || def.outputs.size() != 1 ||
            def.outputs[0].type != SignalType::CaptureStat || def.layout != LayoutRule::Stat) {
            throw Error("Pipeline node {} is a reduction, so it takes a signal and outputs one "
                        "CaptureStat laid out per capture",
                        def.id);
        }
        break;
    case NodeKind::Global:
        if (def.inputs.size() != 1 || def.outputs.size() != 1 || def.passes == 0 ||
            def.inputs[0].type != def.outputs[0].type) {
            throw Error("Pipeline node {} is global, so it takes one signal and outputs one of the "
                        "same type over one or more passes",
                        def.id);
        }
        break;
    case NodeKind::Point:
    case NodeKind::Compact:
        break;
    }
    if (!def.state.empty() && def.kind != NodeKind::Point) {
        throw Error("Pipeline node {} keeps state, which only point nodes do for now", def.id);
    }
    for (const UniformDef& uniform : def.uniforms) {
        if (std::ranges::find(uniform.allowed, uniform.fallback) == uniform.allowed.end()) {
            throw Error("Pipeline node {}'s uniform {} falls back to {}, which it doesn't allow",
                        def.id, uniform.name, uniform.fallback);
        }
    }
}

graph::Published<NodeRegistry>& Registry() {
    static graph::Published<NodeRegistry> registered(BuiltinNodes());
    return registered;
}

}

uint32_t NodeRegistry::Add(NodeDef def) {
    Validate(def);
    const auto found = m_ids.find(def.id);
    if (found != m_ids.end()) {
        m_defs[found->second] = std::move(def);
        return found->second;
    }
    const auto index = uint32_t(m_defs.size());
    m_ids.emplace(def.id, index);
    m_defs.push_back(std::move(def));
    return index;
}

uint32_t NodeRegistry::Find(std::string_view id) const {
    const auto found = m_ids.find(std::string(id));
    return found == m_ids.end() ? kNoDef : found->second;
}

std::shared_ptr<const NodeRegistry> PipelineNodes() {
    return Registry().Get();
}

void RegisterPipelineNode(NodeDef def) {
    Registry().Update([&](NodeRegistry& nodes) { nodes.Add(std::move(def)); });
}

const NodeRegistry& BuiltinNodes() {
    static const NodeRegistry registry = [] {
        NodeRegistry builtins;
        RegisterBuiltinNodes(builtins);
        return builtins;
    }();
    return registry;
}

void RegisterBuiltinNodes(NodeRegistry& registry) {
    registry.Add({.id = "CameraSignal",
                  .kind = NodeKind::Source,
                  .outputs = {{"electronRate", SignalType::ElectronRate}},
                  .uniforms = {{"layout", {"full", "mosaic"}, "full"},
                               {"precision", {"float", "half"}, "float"}},
                  .layout = LayoutRule::Detector});
    registry.Add({.id = "Integrate",
                  .inputs = {{"electronRate", SignalType::ElectronRate}},
                  .outputs = {{"electrons", SignalType::Electrons}},
                  .parameters = {{"time", 1, {0}}},
                  .code = {.module = kCameraNodes, .type = "IntegrateNode"}});
    registry.Add({.id = "Emva1288",
                  .inputs = {{"electrons", SignalType::Electrons}},
                  .outputs = {{"electrons", SignalType::Electrons}},
                  .parameters = {{"time", 1, {0}}},
                  .uniforms = {{"shot", {"poisson", "gaussian", "off"}, "poisson"},
                               {"fixedPatterns", {"false", "true"}, "true"},
                               {"clipping", {"clip", "none"}, "clip"}},
                  .code = {.module = kCameraNodes, .type = "Emva1288Node"}});
    registry.Add({.id = "Adc",
                  .inputs = {{"electrons", SignalType::Electrons}},
                  .outputs = {{"digital", SignalType::Digital}},
                  .code = {.module = kCameraNodes, .type = "AdcNode"}});
    registry.Add({.id = "Gain",
                  .inputs = {{"digital", SignalType::Digital}},
                  .outputs = {{"digital", SignalType::Digital}},
                  .parameters = {{"gain", 1, {1}}},
                  .code = {.module = kCameraNodes, .type = "GainNode"}});
    registry.Add({.id = "ToColor",
                  .inputs = {{"digital", SignalType::Digital}},
                  .outputs = {{"color", SignalType::LinearColor}},
                  .layout = LayoutRule::Color,
                  .code = {.module = kCameraNodes, .type = "ToColorNode"}});
    registry.Add({.id = "Encode",
                  .inputs = {{"color", SignalType::LinearColor}},
                  .outputs = {{"display", SignalType::DisplayColor}},
                  .uniforms = {{"transfer", {"srgb", "rec709"}, "srgb"}},
                  .code = {.module = kCameraNodes, .type = "EncodeNode"}});
    registry.Add({.id = "Demosaic",
                  .kind = NodeKind::Stencil,
                  .inputs = {{"digital", SignalType::Digital}},
                  .outputs = {{"digital", SignalType::Digital}},
                  .uniforms = {{"method", {"bilinear", "malvar"}, "bilinear"}},
                  .radius = 2,
                  .layout = LayoutRule::FullChannels,
                  .code = {.module = kIspNodes, .type = "DemosaicNode"}});
    registry.Add({.id = "GaussianPsf",
                  .kind = NodeKind::Stencil,
                  .inputs = {{"electronRate", SignalType::ElectronRate}},
                  .outputs = {{"electronRate", SignalType::ElectronRate}},
                  .parameters = {{"sigma", 1, {0.7f}}},
                  .radius = 3,
                  .code = {.module = kIspNodes, .type = "GaussianPsfNode"}});
    registry.Add(
        {.id = "DiffractionPsf",
         .kind = NodeKind::Stencil,
         .inputs = {{"electronRate", SignalType::ElectronRate}},
         .outputs = {{"electronRate", SignalType::ElectronRate}},
         .parameters = {{"fNumber", 1, {2.8f}}, {"wavelength", 1, {550}}, {"pixelPitch", 1, {3}}},
         .radius = kMostStencilRadius,
         .code = {.module = kIspNodes, .type = "DiffractionPsfNode"}});
    registry.Add({.id = "Glare",
                  .kind = NodeKind::Global,
                  .inputs = {{"electronRate", SignalType::ElectronRate}},
                  .outputs = {{"electronRate", SignalType::ElectronRate}},
                  .parameters = {{"sigma", 1, {8}}, {"strength", 1, {0.04f}}},
                  .passes = 3,
                  .code = {.module = kIspNodes, .type = "GlareNode"}});
    registry.Add({.id = "Blooming",
                  .kind = NodeKind::Global,
                  .inputs = {{"electrons", SignalType::Electrons}},
                  .outputs = {{"electrons", SignalType::Electrons}},
                  .parameters = {{"reach", 1, {32}, ParameterType::Int}},
                  .code = {.module = kIspNodes, .type = "BloomingNode"}});
    registry.Add({.id = "Denoise",
                  .kind = NodeKind::Stencil,
                  .inputs = {{"color", SignalType::LinearColor}},
                  .outputs = {{"color", SignalType::LinearColor}},
                  .parameters = {{"sigmaSpace", 1, {1}}, {"sigmaRange", 1, {0.05f}}},
                  .radius = 2,
                  .code = {.module = kIspNodes, .type = "DenoiseNode"}});
    registry.Add({.id = "Vignette",
                  .inputs = {{"electronRate", SignalType::ElectronRate}},
                  .outputs = {{"electronRate", SignalType::ElectronRate}},
                  .parameters = {{"strength", 1, {0.3f}}},
                  .code = {.module = kIspNodes, .type = "VignetteNode"}});
    registry.Add({.id = "BlackLevel",
                  .inputs = {{"digital", SignalType::Digital}},
                  .outputs = {{"digital", SignalType::Digital}},
                  .code = {.module = kIspNodes, .type = "BlackLevelNode"}});
    registry.Add(
        {.id = "WhiteBalance",
         .inputs = {{"color", SignalType::LinearColor}, {"stat", SignalType::CaptureStat, true}},
         .outputs = {{"color", SignalType::LinearColor}},
         .parameters = {{"gains", 3, {1, 1, 1}}},
         .code = {.module = kIspNodes, .type = "WhiteBalanceNode"}});
    registry.Add({.id = "Meter",
                  .kind = NodeKind::Reduce,
                  .inputs = {{"color", SignalType::LinearColor}},
                  .outputs = {{"stat", SignalType::CaptureStat}},
                  .parameters = {{"radius", 1, {0.3f}}},
                  .uniforms = {{"mode", {"average", "centerWeighted", "spot"}, "average"}},
                  .layout = LayoutRule::Stat,
                  .code = {.module = kIspNodes, .type = "MeterNode"}});
    registry.Add({.id = "AutoExposure",
                  .inputs = {{"color", SignalType::LinearColor}, {"stat", SignalType::CaptureStat}},
                  .outputs = {{"color", SignalType::LinearColor}},
                  .parameters = {{"target", 1, {0.18f}},
                                 {"speed", 1, {0.5f}},
                                 {"minGain", 1, {1.0f / 64}},
                                 {"maxGain", 1, {64}}},
                  .state = {1},
                  .code = {.module = kIspNodes, .type = "AutoExposureNode"}});
    registry.Add(
        {.id = "Ccm",
         .inputs = {{"color", SignalType::LinearColor}},
         .outputs = {{"color", SignalType::LinearColor}},
         .parameters = {{"red", 3, {1, 0, 0}}, {"green", 3, {0, 1, 0}}, {"blue", 3, {0, 0, 1}}},
         .code = {.module = kIspNodes, .type = "CcmNode"}});
    registry.Add({.id = "LidarReturns",
                  .kind = NodeKind::Source,
                  .outputs = {{"returns", SignalType::Returns}},
                  .layout = LayoutRule::Returns});
    registry.Add({.id = "RangeNoise",
                  .inputs = {{"returns", SignalType::Returns}},
                  .outputs = {{"returns", SignalType::Returns}},
                  .parameters = {{"sigma", 1, {0.02f}}, {"sigmaPerMeter", 1, {0}}},
                  .code = {.module = kLidarNodes, .type = "RangeNoiseNode"}});
    registry.Add({.id = "IntensityNoise",
                  .inputs = {{"returns", SignalType::Returns}},
                  .outputs = {{"returns", SignalType::Returns}},
                  .parameters = {{"sigma", 1, {0}}, {"relative", 1, {0.05f}}},
                  .code = {.module = kLidarNodes, .type = "IntensityNoiseNode"}});
    registry.Add({.id = "Dropout",
                  .inputs = {{"returns", SignalType::Returns}},
                  .outputs = {{"returns", SignalType::Returns}},
                  .parameters = {{"threshold", 1, {0}}, {"rate", 1, {0}}},
                  .code = {.module = kLidarNodes, .type = "DropoutNode"}});
    registry.Add({.id = "RangeQuantize",
                  .inputs = {{"returns", SignalType::Returns}},
                  .outputs = {{"returns", SignalType::Returns}},
                  .parameters = {{"step", 1, {0.01f}}},
                  .code = {.module = kLidarNodes, .type = "RangeQuantizeNode"}});
    registry.Add({.id = "ToPointCloud",
                  .inputs = {{"returns", SignalType::Returns}},
                  .outputs = {{"cloud", SignalType::PointCloud}},
                  .code = {.module = kLidarNodes, .type = "ToPointCloudNode"}});
    registry.Add({.id = "RadarReturns",
                  .kind = NodeKind::Source,
                  .outputs = {{"returns", SignalType::Returns}},
                  .layout = LayoutRule::Returns});
    registry.Add({.id = "RadarNoise",
                  .inputs = {{"returns", SignalType::Returns}},
                  .outputs = {{"returns", SignalType::Returns}},
                  .parameters = {{"rangeSigma", 1, {0.05f}},
                                 {"velocitySigma", 1, {0.1f}},
                                 {"floor", 1, {0}}},
                  .code = {.module = kRadarNodes, .type = "RadarNoiseNode"}});
    registry.Add({.id = "Cfar",
                  .kind = NodeKind::Stencil,
                  .inputs = {{"returns", SignalType::Returns}},
                  .outputs = {{"returns", SignalType::Returns}},
                  .parameters = {{"guard", 1, {1}, ParameterType::Int},
                                 {"training", 1, {2}, ParameterType::Int},
                                 {"falseAlarmRate", 1, {1e-3f}}},
                  .radius = kCfarRadius,
                  .code = {.module = kRadarNodes, .type = "CfarNode"}});
    registry.Add({.id = "Tone",
                  .inputs = {{"color", SignalType::LinearColor}},
                  .outputs = {{"color", SignalType::LinearColor}},
                  .uniforms = {{"curve", {"none", "reinhard", "aces", "agx"}, "none"}},
                  .code = {.module = kIspNodes, .type = "ToneNode"}});
}

}

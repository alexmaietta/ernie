// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/pipeline/builtins.h"

namespace ernie::pipeline {

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
                  .module = kCameraNodes,
                  .type = "IntegrateNode"});
    registry.Add({.id = "Emva1288",
                  .inputs = {{"electrons", SignalType::Electrons}},
                  .outputs = {{"electrons", SignalType::Electrons}},
                  .parameters = {{"time", 1, {0}}},
                  .uniforms = {{"shot", {"poisson", "gaussian", "off"}, "poisson"},
                               {"fixedPatterns", {"false", "true"}, "true"}},
                  .module = kCameraNodes,
                  .type = "Emva1288Node"});
    registry.Add({.id = "Adc",
                  .inputs = {{"electrons", SignalType::Electrons}},
                  .outputs = {{"digital", SignalType::Digital}},
                  .module = kCameraNodes,
                  .type = "AdcNode"});
    registry.Add({.id = "Gain",
                  .inputs = {{"digital", SignalType::Digital}},
                  .outputs = {{"digital", SignalType::Digital}},
                  .parameters = {{"gain", 1, {1}}},
                  .module = kCameraNodes,
                  .type = "GainNode"});
    registry.Add({.id = "ToColor",
                  .inputs = {{"digital", SignalType::Digital}},
                  .outputs = {{"color", SignalType::LinearColor}},
                  .layout = LayoutRule::Color,
                  .module = kCameraNodes,
                  .type = "ToColorNode"});
    registry.Add({.id = "Encode",
                  .inputs = {{"color", SignalType::LinearColor}},
                  .outputs = {{"display", SignalType::DisplayColor}},
                  .uniforms = {{"transfer", {"srgb", "rec709"}, "srgb"}},
                  .module = kCameraNodes,
                  .type = "EncodeNode"});
    registry.Add({.id = "Demosaic",
                  .kind = NodeKind::Stencil,
                  .inputs = {{"digital", SignalType::Digital}},
                  .outputs = {{"digital", SignalType::Digital}},
                  .uniforms = {{"method", {"bilinear"}, "bilinear"}},
                  .radius = 1,
                  .layout = LayoutRule::FullChannels,
                  .module = kIspNodes,
                  .type = "DemosaicNode"});
    registry.Add({.id = "GaussianPsf",
                  .kind = NodeKind::Stencil,
                  .inputs = {{"electronRate", SignalType::ElectronRate}},
                  .outputs = {{"electronRate", SignalType::ElectronRate}},
                  .parameters = {{"sigma", 1, {0.7f}}},
                  .radius = 3,
                  .module = kIspNodes,
                  .type = "GaussianPsfNode"});
    registry.Add({.id = "Denoise",
                  .kind = NodeKind::Stencil,
                  .inputs = {{"color", SignalType::LinearColor}},
                  .outputs = {{"color", SignalType::LinearColor}},
                  .parameters = {{"sigmaSpace", 1, {1}}, {"sigmaRange", 1, {0.05f}}},
                  .radius = 2,
                  .module = kIspNodes,
                  .type = "DenoiseNode"});
    registry.Add({.id = "Vignette",
                  .inputs = {{"electronRate", SignalType::ElectronRate}},
                  .outputs = {{"electronRate", SignalType::ElectronRate}},
                  .parameters = {{"strength", 1, {0.3f}}},
                  .module = kIspNodes,
                  .type = "VignetteNode"});
    registry.Add({.id = "BlackLevel",
                  .inputs = {{"digital", SignalType::Digital}},
                  .outputs = {{"digital", SignalType::Digital}},
                  .module = kIspNodes,
                  .type = "BlackLevelNode"});
    registry.Add(
        {.id = "WhiteBalance",
         .inputs = {{"color", SignalType::LinearColor}, {"stat", SignalType::CaptureStat, true}},
         .outputs = {{"color", SignalType::LinearColor}},
         .parameters = {{"gains", 3, {1, 1, 1}}},
         .module = kIspNodes,
         .type = "WhiteBalanceNode"});
    registry.Add({.id = "Meter",
                  .kind = NodeKind::Reduce,
                  .inputs = {{"color", SignalType::LinearColor}},
                  .outputs = {{"stat", SignalType::CaptureStat}},
                  .parameters = {{"radius", 1, {0.3f}}},
                  .uniforms = {{"mode", {"average", "centerWeighted", "spot"}, "average"}},
                  .layout = LayoutRule::Stat,
                  .module = kIspNodes,
                  .type = "MeterNode"});
    registry.Add({.id = "AutoExposure",
                  .inputs = {{"color", SignalType::LinearColor}, {"stat", SignalType::CaptureStat}},
                  .outputs = {{"color", SignalType::LinearColor}},
                  .parameters = {{"target", 1, {0.18f}},
                                 {"speed", 1, {0.5f}},
                                 {"minGain", 1, {1.0f / 64}},
                                 {"maxGain", 1, {64}}},
                  .state = {1},
                  .module = kIspNodes,
                  .type = "AutoExposureNode"});
    registry.Add(
        {.id = "Ccm",
         .inputs = {{"color", SignalType::LinearColor}},
         .outputs = {{"color", SignalType::LinearColor}},
         .parameters = {{"red", 3, {1, 0, 0}}, {"green", 3, {0, 1, 0}}, {"blue", 3, {0, 0, 1}}},
         .module = kIspNodes,
         .type = "CcmNode"});
    registry.Add({.id = "Tone",
                  .inputs = {{"color", SignalType::LinearColor}},
                  .outputs = {{"color", SignalType::LinearColor}},
                  .uniforms = {{"curve", {"none", "reinhard", "aces"}, "none"}},
                  .module = kIspNodes,
                  .type = "ToneNode"});
}

}

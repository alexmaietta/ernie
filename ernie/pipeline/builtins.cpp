// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/pipeline/builtins.h"

namespace ernie::pipeline {

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
                  .parameters = {{"time"}},
                  .module = kCameraNodes,
                  .type = "IntegrateNode"});
    registry.Add({.id = "Emva1288",
                  .inputs = {{"electrons", SignalType::Electrons}},
                  .outputs = {{"electrons", SignalType::Electrons}},
                  .parameters = {{"time"}},
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
                  .parameters = {{"gain"}},
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
                  .parameters = {{"sigma"}},
                  .radius = 3,
                  .module = kIspNodes,
                  .type = "GaussianPsfNode"});
    registry.Add({.id = "Denoise",
                  .kind = NodeKind::Stencil,
                  .inputs = {{"color", SignalType::LinearColor}},
                  .outputs = {{"color", SignalType::LinearColor}},
                  .parameters = {{"sigmaSpace"}, {"sigmaRange"}},
                  .radius = 2,
                  .module = kIspNodes,
                  .type = "DenoiseNode"});
    registry.Add({.id = "Vignette",
                  .inputs = {{"electronRate", SignalType::ElectronRate}},
                  .outputs = {{"electronRate", SignalType::ElectronRate}},
                  .parameters = {{"strength"}},
                  .module = kIspNodes,
                  .type = "VignetteNode"});
    registry.Add({.id = "BlackLevel",
                  .inputs = {{"digital", SignalType::Digital}},
                  .outputs = {{"digital", SignalType::Digital}},
                  .module = kIspNodes,
                  .type = "BlackLevelNode"});
    registry.Add({.id = "WhiteBalance",
                  .inputs = {{"color", SignalType::LinearColor}},
                  .outputs = {{"color", SignalType::LinearColor}},
                  .parameters = {{"gains", 3}},
                  .module = kIspNodes,
                  .type = "WhiteBalanceNode"});
    registry.Add({.id = "Ccm",
                  .inputs = {{"color", SignalType::LinearColor}},
                  .outputs = {{"color", SignalType::LinearColor}},
                  .parameters = {{"red", 3}, {"green", 3}, {"blue", 3}},
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

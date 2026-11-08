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
}

}

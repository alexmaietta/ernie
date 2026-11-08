// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include "bindings.h"

namespace nb = nanobind;

namespace ernie::python {

namespace {

DetectorPattern FindPattern(std::string_view name) {
    for (uint32_t pattern = 0; pattern < std::size(kDetectorPatternTokens); ++pattern) {
        if (kDetectorPatternTokens[pattern] == name) {
            return DetectorPattern(pattern);
        }
    }
    throw nb::value_error((std::string(name) + " isn't a detector pattern").c_str());
}

nb::dict Ports(const std::vector<PipelineNodePort>& ports) {
    nb::dict found;
    for (const PipelineNodePort& port : ports) {
        found[port.name.c_str()] = nb::make_tuple(port.type, port.optional);
    }
    return found;
}

PipelineRunDesc RunDesc(const Pipeline& pipeline, HeldTensors& held, const nb::dict& inputs,
                        const std::optional<DetectorDesc>& detector, float exposureTime, float iso,
                        uint32_t capture, uint32_t identity, std::optional<uintptr_t> stream) {
    PipelineRunDesc desc{.detector = detector.value_or(DetectorDesc{}),
                         .exposureTime = exposureTime,
                         .iso = iso,
                         .capture = capture,
                         .identity = identity};
    for (auto [key, value] : inputs) {
        const std::string name = nb::cast<std::string>(key);
        desc.inputs[name] = nb::isinstance<device::Tensor>(value)
                                ? nb::cast<device::Tensor>(value)
                                : held.Import(pipeline.GetDevice(), value, stream, name);
    }
    return desc;
}

}

void BindPipeline(nb::module_& module) {
    module.def(
        "pipeline_node",
        [](std::string_view id) -> std::optional<nb::dict> {
            const std::optional<PipelineNodeInfo> info = FindPipelineNode(id);
            if (!info) {
                return std::nullopt;
            }
            nb::dict found;
            found["id"] = info->id;
            found["kind"] = info->kind;
            found["inputs"] = Ports(info->inputs);
            found["outputs"] = Ports(info->outputs);
            return found;
        },
        nb::arg("id"));

    nb::class_<DetectorDesc>(module, "DetectorDesc")
        .def(
            "__init__",
            [](DetectorDesc* desc, std::string_view pattern, float fullWell, float readNoise,
               float darkCurrent, float darkCurrentDoubling, float temperature, float prnu,
               float dsnu, float systemGain, uint32_t bitDepth, float blackLevel, uint32_t seed) {
                new (desc) DetectorDesc{.pattern = FindPattern(pattern),
                                        .fullWell = fullWell,
                                        .readNoise = readNoise,
                                        .darkCurrent = darkCurrent,
                                        .darkCurrentDoubling = darkCurrentDoubling,
                                        .temperature = temperature,
                                        .prnu = prnu,
                                        .dsnu = dsnu,
                                        .systemGain = systemGain,
                                        .bitDepth = bitDepth,
                                        .blackLevel = blackLevel,
                                        .seed = seed};
            },
            nb::kw_only(), nb::arg("pattern") = "rgb", nb::arg("full_well") = 10000.0f,
            nb::arg("read_noise") = 2.0f, nb::arg("dark_current") = 1.0f,
            nb::arg("dark_current_doubling") = 7.0f, nb::arg("temperature") = 293.15f,
            nb::arg("prnu") = 0.005f, nb::arg("dsnu") = 0.5f, nb::arg("system_gain") = 0.25f,
            nb::arg("bit_depth") = 12u, nb::arg("black_level") = 0.0f, nb::arg("seed") = 0u)
        .def_prop_ro("pattern",
                     [](const DetectorDesc& desc) {
                         return std::string(kDetectorPatternTokens[uint32_t(desc.pattern)]);
                     })
        .def_rw("full_well", &DetectorDesc::fullWell)
        .def_rw("read_noise", &DetectorDesc::readNoise)
        .def_rw("dark_current", &DetectorDesc::darkCurrent)
        .def_rw("dark_current_doubling", &DetectorDesc::darkCurrentDoubling)
        .def_rw("temperature", &DetectorDesc::temperature)
        .def_rw("prnu", &DetectorDesc::prnu)
        .def_rw("dsnu", &DetectorDesc::dsnu)
        .def_rw("system_gain", &DetectorDesc::systemGain)
        .def_rw("bit_depth", &DetectorDesc::bitDepth)
        .def_rw("black_level", &DetectorDesc::blackLevel)
        .def_rw("seed", &DetectorDesc::seed);

    nb::class_<PipelineSession>(module, "PipelineSession")
        .def_prop_ro("captures", &PipelineSession::Captures)
        .def(
            "run",
            [](PipelineSession& session, const nb::dict& inputs,
               std::optional<DetectorDesc> detector, float exposureTime, float iso,
               uint32_t identity, std::optional<uintptr_t> stream) {
                HeldTensors held;
                PipelineRunDesc desc = RunDesc(session.GetPipeline(), held, inputs, detector,
                                               exposureTime, iso, 0, identity, stream);
                const nb::gil_scoped_release release;
                return session.Run(std::move(desc));
            },
            nb::arg("inputs"), nb::kw_only(), nb::arg("detector") = nb::none(),
            nb::arg("exposure_time") = 1.0f, nb::arg("iso") = 100.0f, nb::arg("identity") = 0u,
            nb::arg("stream") = nb::none());

    nb::class_<Pipeline>(module, "Pipeline")
        .def(nb::init<const Ref<Scene>&, std::string_view>(), nb::arg("scene"), nb::arg("path"))
        .def_prop_ro("sources", &Pipeline::Sources)
        .def_prop_ro("outputs", &Pipeline::Outputs)
        .def(
            "run",
            [](const Pipeline& pipeline, const nb::dict& inputs,
               std::optional<DetectorDesc> detector, float exposureTime, float iso,
               uint32_t capture, uint32_t identity, std::optional<uintptr_t> stream) {
                HeldTensors held;
                const PipelineRunDesc desc = RunDesc(pipeline, held, inputs, detector, exposureTime,
                                                     iso, capture, identity, stream);
                const nb::gil_scoped_release release;
                return pipeline.Run(desc);
            },
            nb::arg("inputs"), nb::kw_only(), nb::arg("detector") = nb::none(),
            nb::arg("exposure_time") = 1.0f, nb::arg("iso") = 100.0f, nb::arg("capture") = 0u,
            nb::arg("identity") = 0u, nb::arg("stream") = nb::none())
        .def(
            "session", [](const Pipeline& pipeline) { return PipelineSession(pipeline); },
            nb::keep_alive<0, 1>());
}

}

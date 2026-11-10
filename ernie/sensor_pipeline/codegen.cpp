// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/sensor_pipeline/codegen.h"

#include <algorithm>
#include <cstddef>
#include <format>
#include <set>
#include <string_view>

#include "ernie/util/error.h"
#include "ernie/util/text.h"

extern "C" {
extern const unsigned char ernie_sensor_pipeline_module[];
extern const std::size_t ernie_sensor_pipeline_module_size;
extern const unsigned char ernie_sensor_pipeline_elements[];
extern const std::size_t ernie_sensor_pipeline_elements_size;
extern const unsigned char ernie_sensor_pipeline_captures[];
extern const std::size_t ernie_sensor_pipeline_captures_size;
extern const unsigned char ernie_sensor_pipeline_reduce_groups[];
extern const std::size_t ernie_sensor_pipeline_reduce_groups_size;
extern const unsigned char ernie_sensor_pipeline_reduce_partial[];
extern const std::size_t ernie_sensor_pipeline_reduce_partial_size;
}

namespace ernie::sensor_pipeline {

namespace {

std::string NodeType(const NodeDef& def, const ResolvedNode& node) {
    if (node.uniforms.empty()) {
        return def.code.type;
    }
    std::vector<std::string> choices;
    for (const uint32_t choice : node.uniforms) {
        choices.push_back(std::to_string(choice));
    }
    return std::format("{}<{}>", def.code.type, Join(choices));
}

std::string_view Embedded(const unsigned char* bytes, std::size_t size) {
    return {reinterpret_cast<const char*>(bytes), size};
}

std::string_view KernelTemplate(DispatchShape shape) {
    switch (shape) {
    case DispatchShape::Elements:
        return Embedded(ernie_sensor_pipeline_elements, ernie_sensor_pipeline_elements_size);
    case DispatchShape::ReduceGroups:
        return Embedded(ernie_sensor_pipeline_reduce_groups,
                        ernie_sensor_pipeline_reduce_groups_size);
    case DispatchShape::Captures:
        return Embedded(ernie_sensor_pipeline_captures, ernie_sensor_pipeline_captures_size);
    }
    return {};
}

class Generator {
  public:
    Generator(const SensorPipelineDefinition& definition, const NodeRegistry& registry)
        : m_definition(definition), m_pipeline(definition.pipeline), m_registry(registry),
          m_storedSlot(definition.pipeline.values.size(), kNoValue) {
        m_generated.module =
            std::format("ernie.sensor_pipeline.generated.definition{}", definition.id);
        m_generated.path =
            std::format("ernie/sensor_pipeline/generated/definition{}.slang", definition.id);
        for (const StoredValue& stored : definition.segments.stored) {
            m_storedSlot[stored.value] = uint32_t(m_generated.slots.size());
            m_generated.slots.push_back(stored.value);
        }
        for (const ResolvedOutput& output : m_pipeline.outputs) {
            m_generated.slots.push_back(output.value);
        }
    }

    GeneratedSensorPipeline Generate() && {
        AddImports();
        for (uint32_t index = 0; index < m_definition.segments.segments.size(); ++index) {
            switch (m_definition.segments.segments[index].kind) {
            case NodeKind::Reduce:
                AddReduce(index);
                break;
            case NodeKind::Global:
                AddGlobal(index);
                break;
            case NodeKind::Point:
            case NodeKind::Stencil:
                AddFused(index);
                AddUpdates(index);
                break;
            case NodeKind::Source:
            case NodeKind::Compact:
                throw Error("Pipeline passes of compact nodes can't be generated yet");
            }
        }
        AddSourceCopies();
        return std::move(m_generated);
    }

  private:
    const ResolvedNode& Node(uint32_t node) const {
        return m_pipeline.nodes[node];
    }

    const NodeDef& Def(uint32_t node) const {
        return m_registry[m_pipeline.nodes[node].def];
    }

    std::string_view SlangType(uint32_t value) const {
        return Signal(m_pipeline.values[value].type).slangType;
    }

    std::string Context(uint32_t node) const {
        return std::format("PointContext(p, element, {}, {}, {})", node,
                           m_definition.parameterOffsets[node], m_definition.stateOffsets[node]);
    }

    std::vector<uint32_t> OutputSlots(uint32_t value) const {
        std::vector<uint32_t> slots;
        if (m_storedSlot[value] != kNoValue) {
            slots.push_back(m_storedSlot[value]);
        }
        const auto stored = uint32_t(m_definition.segments.stored.size());
        for (uint32_t output = 0; output < m_pipeline.outputs.size(); ++output) {
            if (m_pipeline.outputs[output].value == value) {
                slots.push_back(stored + output);
            }
        }
        return slots;
    }

    void Emit(std::string_view text) {
        m_body += text;
    }

    void BeginKernel(std::string entry, DispatchShape shape) {
        m_body.clear();
        m_generated.kernels.push_back({std::move(entry), shape});
    }

    void EndKernel() {
        const GeneratedKernel& kernel = m_generated.kernels.back();
        m_generated.source +=
            FillTemplate(KernelTemplate(kernel.shape), {{"entry", kernel.entry}, {"body", m_body}});
    }

    void Write(uint32_t value, std::string_view expression) {
        for (const uint32_t slot : OutputSlots(value)) {
            Emit(std::format("    WriteSignal(p, element, {}, {});\n", slot, expression));
        }
    }

    void AddImports() {
        std::set<std::string> modules;
        for (const ResolvedNode& node : m_pipeline.nodes) {
            const NodeDef& def = m_registry[node.def];
            if (def.kind != NodeKind::Source && !def.code.module.empty()) {
                modules.insert(def.code.module);
            }
        }
        std::string imports;
        for (const std::string& module : modules) {
            imports += std::format("import {};\n", module);
        }
        m_generated.source =
            FillTemplate(Embedded(ernie_sensor_pipeline_module, ernie_sensor_pipeline_module_size),
                         {{"imports", imports}});
    }

    void AddReduce(uint32_t index) {
        const uint32_t node = m_definition.segments.segments[index].nodes.front();
        const ResolvedNode& resolved = Node(node);
        const NodeDef& def = Def(node);
        const uint32_t input = resolved.inputs[0];
        if (m_storedSlot[input] == kNoValue) {
            throw Error("{} is a reduction, but its {} isn't stored", resolved.path,
                        def.inputs[0].name);
        }
        const std::string type = NodeType(def, resolved);
        const std::vector<uint32_t> slots = OutputSlots(resolved.outputs[0]);

        BeginKernel(std::format("Segment{}Partial", index), DispatchShape::ReduceGroups);
        Emit(FillTemplate(Embedded(ernie_sensor_pipeline_reduce_partial,
                                   ernie_sensor_pipeline_reduce_partial_size),
                          {{"type", type},
                           {"context", Context(node)},
                           {"signal", SlangType(input)},
                           {"input", std::to_string(m_storedSlot[input])},
                           {"slot", std::to_string(slots.front())}}));
        EndKernel();

        BeginKernel(std::format("Segment{}Finish", index), DispatchShape::Captures);
        Emit(std::format(
            "    const CaptureStat total = {}.Finish({}, SumStatPartials(p, capture, {}));\n", type,
            Context(node), slots.front()));
        for (const uint32_t slot : slots) {
            Emit(std::format("    WriteStat(p, capture, {}, total);\n", slot));
        }
        EndKernel();
    }

    void AddGlobal(uint32_t index) {
        const uint32_t node = m_definition.segments.segments[index].nodes.front();
        const ResolvedNode& resolved = Node(node);
        const NodeDef& def = Def(node);
        const uint32_t input = resolved.inputs[0];
        if (input == kNoValue || m_storedSlot[input] == kNoValue) {
            throw Error("{} is global, but its {} isn't stored", resolved.path, def.inputs[0].name);
        }
        const uint32_t output = resolved.outputs[0];
        for (uint32_t pass = 0; pass < def.passes; ++pass) {
            const uint32_t previous = pass == 0 ? input : resolved.scratch[pass - 1];
            const bool last = pass + 1 == def.passes;
            BeginKernel(std::format("Segment{}Pass{}", index, pass), DispatchShape::Elements);
            Emit(std::format("    {0} result;\n"
                             "    {1}.Evaluate({2}, {3}, Neighborhood<{0}>(p, element, {4}),\n"
                             "        Neighborhood<{0}>(p, element, {5}), result);\n",
                             SlangType(output), NodeType(def, resolved), Context(node), pass,
                             m_storedSlot[input], m_storedSlot[previous]));
            Write(last ? output : resolved.scratch[pass], "result");
            EndKernel();
        }
    }

    void AddFused(uint32_t index) {
        BeginKernel(std::format("Segment{}", index), DispatchShape::Elements);
        std::set<uint32_t> available;
        for (const uint32_t node : m_definition.segments.segments[index].nodes) {
            std::vector<std::string> arguments;
            for (uint32_t port = 0; port < Def(node).inputs.size(); ++port) {
                arguments.push_back(Argument(node, port, available));
            }
            if (Def(node).code.applies) {
                Apply(node, std::move(arguments));
            } else {
                Evaluate(node, std::move(arguments));
            }
            for (const uint32_t value : Node(node).outputs) {
                available.insert(value);
                Write(value, std::format("v{}", value));
            }
        }
        EndKernel();
    }

    std::string Argument(uint32_t node, uint32_t port, std::set<uint32_t>& available) {
        const ResolvedNode& resolved = Node(node);
        const NodeDef& def = Def(node);
        const uint32_t value = resolved.inputs[port];
        const SignalType taken = def.inputs[port].type;
        if (value == kNoValue) {
            return taken == SignalType::CaptureStat
                       ? std::string("CaptureStat.Zero()")
                       : std::format("{}.Make(Channels(0))", Signal(taken).slangType);
        }
        if (def.kind == NodeKind::Stencil && port == 0) {
            if (m_storedSlot[value] == kNoValue) {
                throw Error("{} is a stencil, but its {} isn't stored", resolved.path,
                            def.inputs[port].name);
            }
            Emit(std::format(
                "    const Neighborhood<{0}> n{1} = Neighborhood<{0}>(p, element, {2});\n",
                SlangType(value), value, m_storedSlot[value]));
            return std::format("n{}", value);
        }
        if (!available.contains(value)) {
            if (m_storedSlot[value] == kNoValue) {
                throw Error("{}'s {} reads a value no earlier pass stored", resolved.path,
                            def.inputs[port].name);
            }
            Emit(
                taken == SignalType::CaptureStat
                    ? std::format("    const CaptureStat v{} = ReadStat(p, element.capture, {});\n",
                                  value, m_storedSlot[value])
                    : std::format("    const {0} v{1} = ReadSignal<{0}>(p, element, {2});\n",
                                  SlangType(value), value, m_storedSlot[value]));
            available.insert(value);
        }
        return std::format("v{}", value);
    }

    void Apply(uint32_t node, std::vector<std::string> arguments) {
        const NodeDef& def = Def(node);
        const std::string context = std::format("context{}", node);
        Emit(std::format("    const PointContext {} = {};\n", context, Context(node)));
        Emit(std::format("    {} node{};\n", def.code.type, node));
        uint32_t word = 0;
        for (const ParameterDef& parameter : def.parameters) {
            Emit(std::format("    node{}.{} = {};\n", node, parameter.name,
                             ReadParameter(parameter, context, word)));
            word += parameter.words;
        }
        if (def.code.context != kNoContextArgument) {
            const auto at = std::min(size_t(def.code.context), arguments.size());
            arguments.insert(arguments.begin() + ptrdiff_t(at), context);
        }
        const uint32_t value = Node(node).outputs.front();
        Emit(std::format("    const {} v{} = node{}.Apply({});\n", SlangType(value), value, node,
                         Join(arguments)));
    }

    void Evaluate(uint32_t node, std::vector<std::string> arguments) {
        arguments.insert(arguments.begin(), Context(node));
        for (const uint32_t value : Node(node).outputs) {
            Emit(std::format("    {} v{};\n", SlangType(value), value));
            arguments.push_back(std::format("v{}", value));
        }
        Emit(std::format("    {}.Evaluate({});\n", NodeType(Def(node), Node(node)),
                         Join(arguments)));
    }

    void AddUpdates(uint32_t index) {
        for (const uint32_t node : m_definition.segments.segments[index].nodes) {
            const ResolvedNode& resolved = Node(node);
            const NodeDef& def = Def(node);
            if (def.state.empty()) {
                continue;
            }
            std::vector<std::string> arguments = {Context(node)};
            for (uint32_t port = 0; port < def.inputs.size(); ++port) {
                if (def.inputs[port].type != SignalType::CaptureStat) {
                    continue;
                }
                const uint32_t value = resolved.inputs[port];
                arguments.push_back(value == kNoValue ? std::string("CaptureStat.Zero()")
                                                      : std::format("ReadStat(p, capture, {})",
                                                                    m_storedSlot[value]));
            }
            BeginKernel(std::format("Segment{}Update{}", index, node), DispatchShape::Captures);
            Emit(std::format("    {}.Update({});\n", NodeType(def, resolved), Join(arguments)));
            EndKernel();
        }
    }

    void AddSourceCopies() {
        const auto stored = uint32_t(m_definition.segments.stored.size());
        std::string copies;
        for (uint32_t output = 0; output < m_pipeline.outputs.size(); ++output) {
            const uint32_t value = m_pipeline.outputs[output].value;
            if (Def(m_pipeline.values[value].node).kind == NodeKind::Source) {
                copies += std::format(
                    "    WriteSignal(p, element, {}, ReadSignal<{}>(p, element, {}));\n",
                    stored + output, SlangType(value), m_storedSlot[value]);
            }
        }
        if (!copies.empty()) {
            BeginKernel("CopySources", DispatchShape::Elements);
            Emit(copies);
            EndKernel();
        }
    }

    const SensorPipelineDefinition& m_definition;
    const ResolvedSensorPipeline& m_pipeline;
    const NodeRegistry& m_registry;
    std::vector<uint32_t> m_storedSlot;
    GeneratedSensorPipeline m_generated;
    std::string m_body;
};

}

std::vector<std::string> GeneratedSensorPipeline::Entries() const {
    std::vector<std::string> entries;
    for (const GeneratedKernel& kernel : kernels) {
        entries.push_back(kernel.entry);
    }
    return entries;
}

GeneratedSensorPipeline GenerateSensorPipeline(const SensorPipelineDefinition& definition,
                                               const NodeRegistry& registry) {
    return Generator(definition, registry).Generate();
}

}

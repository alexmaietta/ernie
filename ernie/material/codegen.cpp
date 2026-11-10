// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/material/codegen.h"

#include <algorithm>
#include <cstddef>
#include <format>
#include <string_view>

#include "ernie/util/text.h"

extern "C" {
extern const unsigned char ernie_material_dispatch[];
extern const std::size_t ernie_material_dispatch_size;
extern const unsigned char ernie_material_definition[];
extern const std::size_t ernie_material_definition_size;
extern const unsigned char ernie_material_surface[];
extern const std::size_t ernie_material_surface_size;
extern const unsigned char ernie_material_standard[];
extern const std::size_t ernie_material_standard_size;
extern const unsigned char ernie_material_registered[];
extern const std::size_t ernie_material_registered_size;
}

namespace ernie::material {

namespace {

std::string_view Embedded(const unsigned char* bytes, std::size_t size) {
    return {reinterpret_cast<const char*>(bytes), size};
}

std::string Cases(const MaterialDefinitionSet& set, std::string_view standard,
                  std::string_view surface) {
    std::string cases;
    for (const MaterialDefinition& definition : set.Definitions()) {
        const std::string_view call = definition.surface ? surface : standard;
        cases +=
            std::format("    }} else if (context.definition == {}) {{\n        {}\n", definition.id,
                        std::vformat(call, std::make_format_args(definition.type)));
    }
    return cases;
}

std::string Dispatch(const MaterialDefinitionSet& set) {
    std::string imports;
    std::string linked;
    for (const MaterialDefinition& definition : set.Definitions()) {
        imports += std::format("import {};\n", definition.module);
        linked += std::format(" || definition == {}", definition.id);
    }
    return FillTemplate(
        Embedded(ernie_material_dispatch, ernie_material_dispatch_size),
        {{"imports", imports},
         {"firstGenerated", std::to_string(kFirstGeneratedDefinition)},
         {"linked", linked},
         {"shadeCases", Cases(set, "StandardDefinition<{}, Spectrum>.Shade(context, closure);",
                              "{}<Spectrum>.Shade(context, closure);")},
         {"albedoCases", Cases(set, "return StandardDefinition<{}, Spectrum>.Albedo(context);",
                               "return {}<Spectrum>.Albedo(context);")}});
}

std::string Expression(const ShadingGraph& graph, const GraphLayout& layout, uint32_t node,
                       uint32_t input) {
    const InputBinding& binding = graph.nodes[node].inputs[input];
    const MaterialNodeInput& declared = graph.nodes[node].node->inputs[input];
    const bool spectral = declared.type == MaterialInputType::Spectral;
    if (binding.kind == BindingKind::Pattern) {
        return spectral ? std::format("SpectralValue.Rgb(r{})", binding.source)
                        : std::format("r{}", binding.source);
    }
    const std::string cell = std::format("p.n{}_{}", node, declared.name);
    if (binding.kind == BindingKind::Texture) {
        const std::string sampled =
            std::format("texels.texels[{0}], texels.Handle({0}), &p.textures[{0}]",
                        layout.textures[node][input]);
        if (declared.type == MaterialInputType::Float) {
            return std::format("TexturedFloat({}, {})", cell, sampled);
        }
        const std::string color = std::format("TexturedColor({}, {})", cell, sampled);
        return spectral ? std::format("SpectralValue.Rgb({})", color) : color;
    }
    if (spectral) {
        return std::format("({0}_spectrum != kNoSpectrum ? SpectralValue.Library({0}_spectrum) : "
                           "SpectralValue.Rgb({0}.Vector()))",
                           cell);
    }
    switch (declared.type) {
    case MaterialInputType::Bool:
        return cell + " != 0";
    case MaterialInputType::Float2:
    case MaterialInputType::Float3:
    case MaterialInputType::Float4:
        return cell + ".Vector()";
    default:
        return cell;
    }
}

std::string Call(const ShadingGraph& graph, const GraphLayout& layout, uint32_t node) {
    const MaterialNodeDefinition& definition = *graph.nodes[node].node;
    std::vector<std::string> arguments;
    for (const MaterialNodeParameter& parameter : definition.parameters) {
        switch (parameter.source) {
        case MaterialNodeParameter::Source::Context:
            arguments.emplace_back("context");
            break;
        case MaterialNodeParameter::Source::Point:
            arguments.emplace_back("context.point");
            break;
        case MaterialNodeParameter::Source::Input:
            arguments.push_back(Expression(graph, layout, node, parameter.input));
            break;
        }
    }
    return std::format("        const {} r{} = {}({});\n", InputInfo(definition.result).slangType,
                       node, definition.name, Join(arguments));
}

std::string ParamsFields(const ShadingGraph& graph, const GraphLayout& layout) {
    std::string fields;
    if (layout.textureCount > 0) {
        fields += std::format("    TextureInput textures[{}];\n    MappingRecord mapping;\n",
                              layout.textureCount);
    }
    for (uint32_t node = 0; node < graph.nodes.size(); ++node) {
        for (const MaterialNodeInput& input : graph.nodes[node].node->inputs) {
            fields +=
                std::format("    {} n{}_{};\n", InputInfo(input.type).cellType, node, input.name);
            if (input.type == MaterialInputType::Spectral) {
                fields += std::format("    uint n{}_{}_spectrum;\n", node, input.name);
            }
        }
    }
    return fields;
}

std::string AddImports(const ShadingGraph& graph, MaterialDefinition& module) {
    std::string imports;
    for (const ShadingNode& node : graph.nodes) {
        const MaterialNodeDefinition& definition = *node.node;
        const std::string file = ModulePath(definition.module);
        if (module.files.contains(file)) {
            continue;
        }
        imports += std::format("import {};\n", definition.module);
        module.files.emplace(file, definition.source);
        module.origins.emplace_back(file, definition.file);
        module.origins.emplace_back(definition.module + ".slang", definition.file);
    }
    return imports;
}

std::string Body(const ShadingGraph& graph, const GraphLayout& layout) {
    const auto surface = uint32_t(graph.nodes.size() - 1);
    const MaterialNodeDefinition& top = *graph.nodes[surface].node;
    std::string body;
    if (layout.textureCount > 0) {
        body += std::format("        const SurfaceTexels<{0}> texels = SampleNodeTextures<{0}>("
                            "context, &p.mapping, &p.textures[0]);\n",
                            layout.textureCount);
    }
    for (uint32_t node = 0; node < surface; ++node) {
        body += Call(graph, layout, node);
    }
    body += std::format("        {} node;\n", top.name);
    for (uint32_t input = 0; input < top.inputs.size(); ++input) {
        body += std::format("        node.{} = {};\n", top.inputs[input].name,
                            Expression(graph, layout, surface, input));
    }
    return body;
}

}

bool Texturable(MaterialInputType type) {
    return type == MaterialInputType::Float || type == MaterialInputType::Float3 ||
           type == MaterialInputType::Spectral;
}

GraphLayout LayOut(const ShadingGraph& graph) {
    GraphLayout layout;
    for (const ShadingNode& node : graph.nodes) {
        std::vector<uint32_t>& textures = layout.textures.emplace_back();
        for (const InputBinding& binding : node.inputs) {
            textures.push_back(binding.kind == BindingKind::Texture ? layout.textureCount++
                                                                    : kNoTexture);
        }
    }
    uint32_t offset = 0;
    if (layout.textureCount > 0) {
        layout.mapping = layout.textureCount * kTextureInputBytes;
        offset = layout.mapping + kMappingBytes;
    }
    for (const ShadingNode& node : graph.nodes) {
        std::vector<uint32_t>& cells = layout.cells.emplace_back();
        for (const MaterialNodeInput& input : node.node->inputs) {
            cells.push_back(offset);
            offset += InputInfo(input.type).bytes;
        }
    }
    layout.bytes = offset;
    return layout;
}

graph::ShapeKey MaterialDefinitionKey(const ShadingGraph& graph) {
    graph::ShapeWriter writer;
    writer.Add(uint32_t(graph.nodes.size()));
    for (const ShadingNode& node : graph.nodes) {
        writer.Add(node.node->module);
        writer.Add(node.node->name);
        writer.Add(uint32_t(node.inputs.size()));
        for (const InputBinding& binding : node.inputs) {
            writer.Add(uint32_t(binding.kind));
            if (binding.kind == BindingKind::Pattern) {
                writer.Add(binding.source);
            }
        }
    }
    return std::move(writer).Finish();
}

MaterialDefinition GenerateMaterialDefinition(uint32_t id, const ShadingGraph& graph) {
    const GraphLayout layout = LayOut(graph);
    const MaterialNodeDefinition& top = *graph.nodes.back().node;
    MaterialDefinition module{.id = id,
                              .type = std::format("Definition{}", id),
                              .module = std::format("ernie.material.generated.d{}", id),
                              .words = layout.bytes / uint32_t(sizeof(uint32_t)),
                              .surface = top.role == MaterialNodeRole::Surface};
    const std::string& type = module.type;
    const std::string imports = AddImports(graph, module);
    const std::string fields = ParamsFields(graph, layout);
    const std::string body = Body(graph, layout);
    const std::string definition = FillTemplate(
        module.surface ? Embedded(ernie_material_surface, ernie_material_surface_size)
                       : Embedded(ernie_material_standard, ernie_material_standard_size),
        {{"type", type}, {"body", body}});
    module.source =
        FillTemplate(Embedded(ernie_material_definition, ernie_material_definition_size),
                     {{"node", top.name},
                      {"imports", imports},
                      {"type", type},
                      {"fields", fields},
                      {"definition", definition}});
    return module;
}

std::map<std::string, std::string, std::less<>>
ClosurePrimitiveSources(std::span<const std::shared_ptr<const ClosurePrimitive>> primitives) {
    std::map<std::string, std::string, std::less<>> sources;
    if (primitives.empty()) {
        return sources;
    }
    std::string imports;
    std::string kinds;
    std::string glossy;
    std::string estimate;
    std::string evaluate;
    std::string pdf;
    std::string direction;
    for (const std::shared_ptr<const ClosurePrimitive>& primitive : primitives) {
        sources[ModulePath(primitive->module)] = primitive->source;
        imports += std::format("import {};\n", primitive->module);
        kinds += std::format("static const uint kLobe{} = {};\n", primitive->name, primitive->kind);
        const auto add = [&](std::string& cases, std::string_view call) {
            cases += std::format("    case {}:\n        return {}.{};\n", primitive->kind,
                                 primitive->name, call);
        };
        add(glossy, "Glossy()");
        add(estimate, "Estimate<Spectrum>(tint, alpha, params.values, outgoing)");
        add(evaluate, "Evaluate<Spectrum>(tint, alpha, params.values, outgoing, incoming)");
        add(pdf, "Pdf(alpha, params.values, outgoing, incoming)");
        add(direction, "Direction(alpha, params.values, outgoing, u)");
    }
    sources[kRegisteredFile] =
        FillTemplate(Embedded(ernie_material_registered, ernie_material_registered_size),
                     {{"imports", imports},
                      {"firstLobe", std::to_string(kFirstPrimitiveLobe)},
                      {"kinds", kinds},
                      {"glossy", glossy},
                      {"estimate", estimate},
                      {"evaluate", evaluate},
                      {"pdf", pdf},
                      {"direction", direction}});
    return sources;
}

std::map<std::string, std::string, std::less<>>
MaterialDispatchSources(const MaterialDefinitionSet& set) {
    std::map<std::string, std::string, std::less<>> sources(set.Files().begin(), set.Files().end());
    sources[kDispatchFile] = Dispatch(set);
    for (const MaterialDefinition& definition : set.Definitions()) {
        sources[ModulePath(definition.module)] = definition.source;
        for (const auto& [path, source] : definition.files) {
            sources[path] = source;
        }
    }
    return sources;
}

MaterialDefinitionSet::MaterialDefinitionSet(std::vector<MaterialDefinition> definitions,
                                             std::map<std::string, std::string, std::less<>> files)
    : m_definitions(std::move(definitions)), m_files(std::move(files)), m_hash(kFnvBasis) {
    for (const auto& [path, source] : m_files) {
        m_hash = Fnv1a(path, m_hash);
        m_hash = Fnv1a(source, m_hash);
    }
    std::ranges::sort(m_definitions, {}, &MaterialDefinition::id);
    for (const MaterialDefinition& definition : m_definitions) {
        m_hash = Fnv1a(std::to_string(definition.id), m_hash);
        m_hash = Fnv1a(definition.type, m_hash);
        m_hash = Fnv1a(definition.source, m_hash);
        for (const auto& [path, source] : definition.files) {
            m_hash = Fnv1a(path, m_hash);
            m_hash = Fnv1a(source, m_hash);
        }
    }
}

bool MaterialDefinitionSet::Contains(uint32_t id) const {
    return id < kFirstGeneratedDefinition ||
           std::ranges::binary_search(m_definitions, id, {}, &MaterialDefinition::id);
}

}

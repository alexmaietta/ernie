// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/materials.h"

#include <set>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <string_view>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/graph/graph.h"
#include "ernie/material/codegen.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/connections.h"
#include "ernie/scene/tables/kernels/materials.h"
#include "ernie/scene/tables/textures.h"
#include "ernie/stage/publish/change_set.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {

namespace {

constexpr uint32_t Word(size_t offset) {
    return uint32_t(offset / sizeof(uint32_t));
}

constexpr std::string_view kPreviewInputNames[kPreviewInputs] = {
    "diffuseColor", "emissiveColor",      "specularColor", "metallic",  "roughness",
    "clearcoat",    "clearcoatRoughness", "opacity",       "occlusion", "normal",
    "detailNormal", "detailAlbedo",       "displacement",
};

constexpr Float4 kItuMaterials[] = {
    {1, 0, 0, 0},
    {5.24f, 0, 0.0462f, 0.7822f},
    {3.91f, 0, 0.0238f, 0.16f},
    {2.73f, 0, 0.0085f, 0.9395f},
    {1.99f, 0, 0.0047f, 1.0718f},
    {6.31f, 0, 0.0036f, 1.3394f},
    {1.48f, 0, 0.0011f, 1.0750f},
    {2.58f, 0, 0.0217f, 0.7800f},
    {2.71f, 0, 0.33f, 0},
    {7.074f, 0, 0.0055f, 0.9262f},
    {3.66f, 0, 0.0044f, 1.3515f},
    {1, 0, 1e7f, 0},
    {3, 0, 0.00015f, 2.52f},
    {15, -0.1f, 0.035f, 1.63f},
    {30, -0.4f, 0.15f, 1.30f},
};

constexpr float kVisibleFirstNm = 380;
constexpr float kVisibleLastNm = 780;

constexpr bool Spectral(PreviewInput input) {
    return input == PreviewInput::DiffuseColor || input == PreviewInput::EmissiveColor ||
           input == PreviewInput::SpecularColor;
}

constexpr bool ColorInput(uint32_t input) {
    return input == uint32_t(PreviewInput::DiffuseColor) ||
           input == uint32_t(PreviewInput::EmissiveColor) ||
           input == uint32_t(PreviewInput::SpecularColor) ||
           input == uint32_t(PreviewInput::Normal) ||
           input == uint32_t(PreviewInput::DetailNormal) ||
           input == uint32_t(PreviewInput::DetailAlbedo);
}

bool Channel(std::string_view output, bool color, TextureChannel& channel) {
    if (output == "rgb") {
        channel = color ? TextureChannel::Rgb : TextureChannel::R;
    } else if (output == "r") {
        channel = TextureChannel::R;
    } else if (output == "g") {
        channel = TextureChannel::G;
    } else if (output == "b") {
        channel = TextureChannel::B;
    } else if (output == "a") {
        channel = TextureChannel::A;
    } else {
        return false;
    }
    return true;
}

std::vector<ColumnRef> MaterialInputs() {
    std::vector<ColumnRef> inputs = {kPlaneAxis, kDetectorAPIDetectorSpectralRange,
                                     kThermalAPIThermalTemperature};
    for (const SchemaId schema :
         {SchemaId::Connectable, SchemaId::UsdPreviewSurface, SchemaId::SurfaceDetailAPI,
          SchemaId::UsdUVTexture, SchemaId::UsdPrimvarReader_float2, SchemaId::UsdTransform2d,
          SchemaId::TexturePlacement, SchemaId::ItuMaterial, SchemaId::CheckerboardAPI}) {
        const Span<const ColumnInfo> columns = Columns(schema);
        for (uint32_t index = 0; index < columns.size(); ++index) {
            inputs.push_back({schema, &columns[index], index});
        }
    }
    return inputs;
}

}

bool MaterialTable::Restructured(const ChangeSet& changes) const {
    if (changes.resync) {
        return true;
    }
    for (const ColumnRef column : {kConnectableConnectionsSources, kConnectableConnectionsInputs,
                                   kConnectableConnectionsOutputs}) {
        if (!changes.ArrayRows(column).empty()) {
            return true;
        }
    }
    const Prims& prims = m_structure.GetPrims();
    const auto touched = [&](uint32_t prim) { return prim < m_touched.size() && m_touched[prim]; };
    for (const StructureEntry& entry : changes.added) {
        if (touched(entry.prim) || (prims.Alive(entry.prim) &&
                                    IsA(prims.Record(entry.prim).schema, SchemaId::Connectable))) {
            return true;
        }
    }
    for (const StructureEntry& entry : changes.removed) {
        if (touched(entry.prim)) {
            return true;
        }
    }
    for (const StructureEntry& entry : changes.apis) {
        if (touched(entry.prim) || entry.api == SchemaId::SurfaceDetailAPI) {
            return true;
        }
    }
    return false;
}

bool MaterialTable::Changed(SchemaId schema) const {
    const ColumnStore& store = m_structure.GetColumnStore();
    return std::ranges::any_of(Inputs(), [&](ColumnRef column) {
        return column.schema == schema && store.LastWrite(column) > m_writes;
    });
}

std::span<const ColumnRef> MaterialTable::Inputs() const {
    if (m_inputSchemas != SchemaCount()) {
        m_inputs = MaterialInputs();
        for (uint32_t index = uint32_t(SchemaId::Count); index < SchemaCount(); ++index) {
            const SchemaId schema = SchemaId(index);
            const Span<const ColumnInfo> columns = Columns(schema);
            for (uint32_t column = 0; column < columns.size(); ++column) {
                m_inputs.push_back({schema, &columns[column], column});
            }
        }
        m_inputSchemas = SchemaCount();
    }
    return m_inputs;
}

void MaterialTable::Describe(RenderTables& tables) const {
    tables.materials.address = m_records.Buffer() ? m_records.Buffer()->Address() : 0;
    tables.materialWords.address = m_words.Address();
    tables.checkers.address = m_checkers.Address();
    tables.temperatures.address = m_temperatures.Address();
    tables.radios.address = m_radios.Buffer() ? m_radios.Buffer()->Address() : 0;
    tables.temperaturePrimvar = m_temperaturePrimvar;
}

std::span<const MaterialTable::Connection> MaterialTable::Connections(uint32_t prim) {
    const std::span<const Connection> connections = m_connections.Get(prim);
    if (prim < m_touched.size()) {
        m_touched[prim] = true;
    }
    for (const Connection& connection : connections) {
        if (connection.source < m_touched.size()) {
            m_touched[connection.source] = true;
        }
    }
    return connections;
}

void MaterialTable::CopyCell(ColumnRef column, uint32_t prim, uint32_t destination) {
    m_pack.Copy(column, m_structure.GetPrims().Row(prim, column.schema), destination);
}

void MaterialTable::ResolvePlacement(uint32_t placement, uint32_t mapping) {
    const Prims& prims = m_structure.GetPrims();
    const auto at = [&](size_t offset) { return mapping + Word(offset); };
    const auto literal = [&](size_t offset, uint32_t value) { m_pack.Literal(at(offset), value); };
    const auto real = [&](size_t offset, float value) {
        literal(offset, std::bit_cast<uint32_t>(value));
    };
    const SchemaId schema = placement == kNoPrim ? SchemaId::None : prims.Record(placement).schema;
    if (schema == SchemaId::TexturePlacement) {
        CopyCell(kTexturePlacementProjection, placement, at(offsetof(MappingRecord, projection)));
        CopyCell(kTexturePlacementProjectionSpace, placement,
                 at(offsetof(MappingRecord, projectionSpace)));
        CopyCell(kTexturePlacementUvPrimvar, placement, at(offsetof(MappingRecord, uvPrimvar)));
        CopyCell(kTexturePlacementTileSize, placement, at(offsetof(MappingRecord, tileSize)));
        CopyCell(kTexturePlacementTriplanarBlend, placement,
                 at(offsetof(MappingRecord, triplanarBlend)));
        CopyCell(kTexturePlacementUvScale, placement, at(offsetof(MappingRecord, uvScale)));
        CopyCell(kTexturePlacementUvOffset, placement, at(offsetof(MappingRecord, uvOffset)));
        CopyCell(kTexturePlacementUvRotation, placement, at(offsetof(MappingRecord, uvRotation)));
        CopyCell(kTexturePlacementHexTiling, placement, at(offsetof(MappingRecord, hexTiling)));
        CopyCell(kTexturePlacementHexTilingContrast, placement,
                 at(offsetof(MappingRecord, hexTilingContrast)));
        CopyCell(kTexturePlacementHexTilingRotation, placement,
                 at(offsetof(MappingRecord, hexTilingRotation)));
        return;
    }
    literal(offsetof(MappingRecord, projection), 0);
    literal(offsetof(MappingRecord, projectionSpace), 0);
    real(offsetof(MappingRecord, tileSize), 1);
    real(offsetof(MappingRecord, triplanarBlend), 0.25f);
    literal(offsetof(MappingRecord, hexTiling), 0);
    real(offsetof(MappingRecord, hexTilingContrast), 0.5f);
    real(offsetof(MappingRecord, hexTilingRotation), 180);
    uint32_t reader = placement;
    if (schema == SchemaId::UsdTransform2d) {
        CopyCell(kUsdTransform2dScale, placement, at(offsetof(MappingRecord, uvScale)));
        CopyCell(kUsdTransform2dTranslation, placement, at(offsetof(MappingRecord, uvOffset)));
        CopyCell(kUsdTransform2dRotation, placement, at(offsetof(MappingRecord, uvRotation)));
        reader = kNoPrim;
        for (const Connection& connection : Connections(placement)) {
            if (connection.input == "in") {
                reader = connection.source;
            }
        }
    } else {
        real(offsetof(MappingRecord, uvScale), 1);
        real(offsetof(MappingRecord, uvScale) + sizeof(float), 1);
        real(offsetof(MappingRecord, uvOffset), 0);
        real(offsetof(MappingRecord, uvOffset) + sizeof(float), 0);
        real(offsetof(MappingRecord, uvRotation), 0);
    }
    if (reader != kNoPrim && prims.Record(reader).schema == SchemaId::UsdPrimvarReader_float2) {
        CopyCell(kUsdPrimvarReader_float2Varname, reader, at(offsetof(MappingRecord, uvPrimvar)));
    } else {
        if (reader != kNoPrim) {
            Log(LogLevel::Warning,
                "{} reads its coordinates from {}, which isn't a UsdPrimvarReader_float2; it "
                "reads the st primvar instead",
                m_structure.GetPaths().Path(placement), m_structure.GetPaths().Path(reader));
        }
        literal(offsetof(MappingRecord, uvPrimvar), m_structure.GetTokens().Intern("st").id);
    }
}

void MaterialTable::ResolvePreview(uint32_t material, uint32_t shader) {
    const Prims& prims = m_structure.GetPrims();
    const Paths& paths = m_structure.GetPaths();
    const uint32_t base = m_wordCount;
    m_wordCount += Word(sizeof(PreviewParams));
    const auto at = [&](size_t offset) { return base + Word(offset); };
    CopyCell(kUsdPreviewSurfaceDiffuseColor, shader, at(offsetof(PreviewParams, diffuseColor)));
    CopyCell(kUsdPreviewSurfaceUseSpecularWorkflow, shader,
             at(offsetof(PreviewParams, useSpecularWorkflow)));
    CopyCell(kUsdPreviewSurfaceEmissiveColor, shader, at(offsetof(PreviewParams, emissiveColor)));
    CopyCell(kUsdPreviewSurfaceMetallic, shader, at(offsetof(PreviewParams, metallic)));
    CopyCell(kUsdPreviewSurfaceSpecularColor, shader, at(offsetof(PreviewParams, specularColor)));
    CopyCell(kUsdPreviewSurfaceRoughness, shader, at(offsetof(PreviewParams, roughness)));
    CopyCell(kUsdPreviewSurfaceIor, shader, at(offsetof(PreviewParams, ior)));
    CopyCell(kUsdPreviewSurfaceClearcoat, shader, at(offsetof(PreviewParams, clearcoat)));
    CopyCell(kUsdPreviewSurfaceClearcoatRoughness, shader,
             at(offsetof(PreviewParams, clearcoatRoughness)));
    CopyCell(kUsdPreviewSurfaceOpacity, shader, at(offsetof(PreviewParams, opacity)));
    const size_t detail = offsetof(PreviewParams, detail);
    CopyCell(kSurfaceDetailAPINormal, shader, at(detail + offsetof(DetailRecord, normal)));
    CopyCell(kSurfaceDetailAPINormalStrength, shader,
             at(detail + offsetof(DetailRecord, normalStrength)));
    CopyCell(kSurfaceDetailAPIDetailNormal, shader,
             at(detail + offsetof(DetailRecord, detailNormal)));
    CopyCell(kSurfaceDetailAPIDetailNormalStrength, shader,
             at(detail + offsetof(DetailRecord, detailNormalStrength)));
    CopyCell(kSurfaceDetailAPIDetailAlbedo, shader,
             at(detail + offsetof(DetailRecord, detailAlbedo)));
    CopyCell(kSurfaceDetailAPIDetailAlbedoStrength, shader,
             at(detail + offsetof(DetailRecord, detailAlbedoStrength)));
    CopyCell(kSurfaceDetailAPIDetailScale, shader,
             at(detail + offsetof(DetailRecord, detailScale)));
    CopyCell(kSurfaceDetailAPIDetailOffset, shader,
             at(detail + offsetof(DetailRecord, detailOffset)));
    CopyCell(kSurfaceDetailAPIDetailRotation, shader,
             at(detail + offsetof(DetailRecord, detailRotation)));
    CopyCell(kSurfaceDetailAPIDetailHexTiling, shader,
             at(detail + offsetof(DetailRecord, detailHexTiling)));
    CopyCell(kSurfaceDetailAPIEnableDetail, shader,
             at(detail + offsetof(DetailRecord, enableDetail)));

    const std::span<const Connection> connections = Connections(shader);
    uint32_t placement = kNoPrim;
    bool placed = false;
    uint32_t textured = 0;
    for (uint32_t input = 0; input < kPreviewInputs; ++input) {
        const size_t slot = offsetof(PreviewParams, inputs) + input * sizeof(TextureInput);
        const auto unbound = [&] {
            m_pack.Literal(at(slot + offsetof(TextureInput, handle)), 0);
            m_pack.Literal(at(slot + offsetof(TextureInput, handle)) + 1, 0);
        };
        const auto found =
            std::ranges::find(connections, kPreviewInputNames[input], &Connection::input);
        const bool spectral = found != connections.end() && found->source != kNoPrim &&
                              prims.Record(found->source).schema == SchemaId::SampledSpectrum;
        uint32_t spectrum = kNoSpectrum;
        if (spectral && Spectral(PreviewInput(input))) {
            spectrum = prims.Row(found->source, SchemaId::SampledSpectrum);
            m_spectral.insert(material);
        } else if (spectral) {
            Log(LogLevel::Warning,
                "{}'s {} connects to the spectrum {}, but only diffuseColor, emissiveColor and "
                "specularColor take spectra; the input keeps its value",
                m_structure.GetPaths().Path(shader), kPreviewInputNames[input],
                m_structure.GetPaths().Path(found->source));
        }
        m_pack.Literal(at(slot + offsetof(TextureInput, spectrum)), spectrum);
        if (found == connections.end() || spectral) {
            unbound();
            continue;
        }
        const uint32_t texture = found->source;
        TextureChannel channel = TextureChannel::Rgb;
        if (!BindTexture(shader, *found, kPreviewInputNames[input], ColorInput(input), at(slot),
                         channel)) {
            unbound();
            continue;
        }
        ++textured;
        if (input == uint32_t(PreviewInput::DetailNormal) ||
            input == uint32_t(PreviewInput::DetailAlbedo)) {
            continue;
        }
        uint32_t coordinates = kNoPrim;
        for (const Connection& connection : Connections(texture)) {
            if (connection.input == "st") {
                coordinates = connection.source;
            }
        }
        if (!placed) {
            placement = coordinates;
            placed = true;
        } else if (coordinates != placement) {
            Log(LogLevel::Warning,
                "{}'s textures read coordinates from more than one source; {} reads them through "
                "the placement of its other textures",
                paths.Path(shader), paths.Path(texture));
        }
    }
    ResolvePlacement(placement, at(offsetof(PreviewParams, mapping)));
    Define(material, textured > 0 ? kDefinitionPreviewTextured : kDefinitionPreviewConstant, base);
    if (GetLogLevel() <= LogLevel::Debug) {
        Log(LogLevel::Debug, "{} shades with {} through {}, with {} textures in {} words",
            paths.Path(material),
            textured > 0 ? "a textured UsdPreviewSurface" : "UsdPreviewSurface", paths.Path(shader),
            textured, Word(sizeof(PreviewParams)));
    }
}

bool MaterialTable::BindTexture(uint32_t shader, const Connection& connection,
                                std::string_view input, bool color, uint32_t slot,
                                TextureChannel& channel) {
    const Prims& prims = m_structure.GetPrims();
    const Paths& paths = m_structure.GetPaths();
    const uint32_t texture = connection.source;
    if (texture == kNoPrim || prims.Record(texture).schema != SchemaId::UsdUVTexture) {
        Log(LogLevel::Warning,
            "{}'s {} connects to {}, which isn't a UsdUVTexture; the input keeps its value",
            paths.Path(shader), input,
            texture == kNoPrim ? std::string("nothing") : paths.Path(texture));
        return false;
    }
    if (!Channel(connection.output, color, channel)) {
        Log(LogLevel::Warning,
            "{}'s {} reads output {} of {}, which isn't rgb, r, g, b or a; the input keeps "
            "its value",
            paths.Path(shader), input, connection.output, paths.Path(texture));
        return false;
    }
    if (slot == kNoPrim) {
        return true;
    }
    const uint32_t handle = slot + Word(offsetof(TextureInput, handle));
    m_bindings.push_back({texture, m_pack.Literal(handle, 0), m_pack.Literal(handle + 1, 0)});
    m_pack.Literal(slot + Word(offsetof(TextureInput, channel)), uint32_t(channel));
    CopyCell(kUsdUVTextureScale, texture, slot + Word(offsetof(TextureInput, scale)));
    CopyCell(kUsdUVTextureBias, texture, slot + Word(offsetof(TextureInput, bias)));
    return true;
}

void MaterialTable::Define(uint32_t material, uint32_t definition, uint32_t words) {
    m_records.Set(material, {.definition = definition,
                             .words = words,
                             .previousDefinition = kDefinitionPlaceholder,
                             .previousWords = 0});
}

std::vector<MaterialTable::Reached>
MaterialTable::ReachNodes(uint32_t shader,
                          std::shared_ptr<const material::MaterialNodeDefinition> surface) {
    const Prims& prims = m_structure.GetPrims();
    const Paths& paths = m_structure.GetPaths();
    std::vector<Reached> reached = {{.prim = shader, .node = std::move(surface)}};
    for (uint32_t k = 0; k < reached.size(); ++k) {
        const uint32_t prim = reached[k].prim;
        const std::shared_ptr<const material::MaterialNodeDefinition> node = reached[k].node;
        const std::span<const Connection> connections = Connections(prim);
        std::vector<material::InputBinding> inputs(node->inputs.size());
        std::vector<Connection> textures(node->inputs.size());
        std::vector<uint32_t> spectra(node->inputs.size(), kNoPrim);
        for (uint32_t i = 0; i < node->inputs.size(); ++i) {
            const material::MaterialNodeInput& input = node->inputs[i];
            const auto found = std::ranges::find(connections, input.name, &Connection::input);
            if (found == connections.end()) {
                continue;
            }
            const uint32_t source = found->source;
            const SchemaId schema =
                source == kNoPrim ? SchemaId::None : prims.Record(source).schema;
            std::shared_ptr<const material::MaterialNodeDefinition> pattern =
                IsRuntime(schema) ? material::FindMaterialNode(Info(schema).name) : nullptr;
            if (schema == SchemaId::SampledSpectrum) {
                if (input.type == material::MaterialInputType::Spectral) {
                    spectra[i] = source;
                } else {
                    Log(LogLevel::Warning,
                        "{}'s {} connects to the spectrum {}, but only SpectralValue inputs take "
                        "spectra; the input keeps its value",
                        paths.Path(prim), input.name, paths.Path(source));
                }
                continue;
            }
            const material::MaterialInputType fed =
                input.type == material::MaterialInputType::Spectral
                    ? material::MaterialInputType::Float3
                    : input.type;
            if (pattern && pattern->role == material::MaterialNodeRole::Pattern) {
                if (pattern->result != fed) {
                    Log(LogLevel::Warning,
                        "{}'s {} connects to {}, whose result has another type; the input keeps "
                        "its value",
                        paths.Path(prim), input.name, paths.Path(source));
                    continue;
                }
                auto at = std::ranges::find(reached, source, &Reached::prim);
                if (at == reached.end()) {
                    reached.push_back({.prim = source, .node = std::move(pattern)});
                    at = reached.end() - 1;
                }
                inputs[i] = {material::BindingKind::Pattern, uint32_t(at - reached.begin())};
                continue;
            }
            if (!material::Texturable(input.type)) {
                Log(LogLevel::Warning,
                    "{}'s {} connects to {}, but only float and float3 inputs take textures; "
                    "the input keeps its value",
                    paths.Path(prim), input.name,
                    source == kNoPrim ? std::string("nothing") : paths.Path(source));
                continue;
            }
            TextureChannel channel = TextureChannel::Rgb;
            if (BindTexture(prim, *found, input.name,
                            input.type != material::MaterialInputType::Float, kNoPrim, channel)) {
                inputs[i] = {material::BindingKind::Texture, 0};
                textures[i] = *found;
            }
        }
        reached[k].inputs = std::move(inputs);
        reached[k].textures = std::move(textures);
        reached[k].spectra = std::move(spectra);
    }
    return reached;
}

std::vector<uint32_t> MaterialTable::OrderPatterns(std::vector<Reached>& reached) const {
    const Paths& paths = m_structure.GetPaths();
    std::vector<uint32_t> order;
    while (true) {
        graph::Graph patterns;
        for (size_t node = 0; node < reached.size(); ++node) {
            patterns.Add(0);
        }
        for (uint32_t k = 0; k < reached.size(); ++k) {
            for (uint32_t i = 0; i < reached[k].inputs.size(); ++i) {
                if (reached[k].inputs[i].kind == material::BindingKind::Pattern) {
                    patterns.Connect(k, i, reached[k].inputs[i].source, 0);
                }
            }
        }
        const graph::Ordering ordering = patterns.Order();
        if (ordering.cycle.empty()) {
            order = ordering.order;
            break;
        }
        const uint32_t closing = ordering.cycle.back();
        for (material::InputBinding& binding : reached[closing].inputs) {
            if (binding.kind == material::BindingKind::Pattern &&
                binding.source == ordering.cycle.front()) {
                binding = {};
            }
        }
        Log(LogLevel::Warning, "{} feeds back into itself through {}; that input keeps its value",
            paths.Path(reached[ordering.cycle.front()].prim), paths.Path(reached[closing].prim));
    }
    order.resize(size_t(std::ranges::find(order, 0u) - order.begin()) + 1);
    return order;
}

void MaterialTable::ResolveNode(uint32_t material, uint32_t shader,
                                std::shared_ptr<const material::MaterialNodeDefinition> surface) {
    const Prims& prims = m_structure.GetPrims();
    const Paths& paths = m_structure.GetPaths();
    std::vector<Reached> reached = ReachNodes(shader, std::move(surface));
    const std::vector<uint32_t> order = OrderPatterns(reached);
    std::vector<uint32_t> position(reached.size(), kNoPrim);
    material::ShadingGraph graph;
    for (const uint32_t k : order) {
        position[k] = uint32_t(graph.nodes.size());
        graph.nodes.push_back({reached[k].node, reached[k].inputs});
    }
    for (material::ShadingNode& node : graph.nodes) {
        for (material::InputBinding& binding : node.inputs) {
            if (binding.kind == material::BindingKind::Pattern) {
                binding.source = position[binding.source];
            }
        }
    }
    const material::GraphLayout layout = material::LayOut(graph);
    const uint32_t definition =
        m_library.Intern(material::MaterialDefinitionKey(graph), [&](uint32_t id) {
            material::MaterialDefinition module = material::GenerateMaterialDefinition(id, graph);
            Log(LogLevel::Debug,
                "new material definition {} for {} with {}, {} in {} words and {} bytes of "
                "generated "
                "Slang",
                id, graph.nodes.back().node->name, Counted(graph.nodes.size() - 1, "pattern"),
                Counted(layout.textureCount, "texture"), Word(layout.bytes), module.source.size());
            return module;
        });
    m_wordCount = (m_wordCount + 1) & ~1u;
    const uint32_t base = m_wordCount;
    m_wordCount += Word(layout.bytes);
    uint32_t coordinates = kNoPrim;
    bool placed = false;
    for (uint32_t n = 0; n < order.size(); ++n) {
        const Reached& node = reached[order[n]];
        const SchemaId schema = SchemaId(node.node->schema);
        const Span<const ColumnInfo> columns = Columns(schema);
        for (uint32_t i = 0; i < node.node->inputs.size(); ++i) {
            CopyCell({schema, &columns[i], i}, node.prim, base + Word(layout.cells[n][i]));
            if (node.node->inputs[i].type == material::MaterialInputType::Spectral) {
                uint32_t spectrum = kNoSpectrum;
                if (node.spectra[i] != kNoPrim) {
                    spectrum = prims.Row(node.spectra[i], SchemaId::SampledSpectrum);
                    m_spectral.insert(material);
                }
                m_pack.Literal(base + Word(layout.cells[n][i] + 12), spectrum);
            }
            if (node.inputs[i].kind != material::BindingKind::Texture) {
                continue;
            }
            TextureChannel channel = TextureChannel::Rgb;
            BindTexture(node.prim, node.textures[i], node.node->inputs[i].name,
                        node.node->inputs[i].type != material::MaterialInputType::Float,
                        base + Word(layout.textures[n][i] * material::kTextureInputBytes), channel);
            uint32_t read = kNoPrim;
            for (const Connection& connection : Connections(node.textures[i].source)) {
                if (connection.input == "st") {
                    read = connection.source;
                }
            }
            if (!placed) {
                coordinates = read;
                placed = true;
            }
        }
    }
    if (layout.textureCount > 0) {
        ResolvePlacement(coordinates, base + Word(layout.mapping));
    }
    Define(material, definition, base);
    if (GetLogLevel() <= LogLevel::Debug) {
        Log(LogLevel::Debug, "{} shades with node {} through {} as definition {}, in {} words",
            paths.Path(material), graph.nodes.back().node->name, paths.Path(shader), definition,
            Word(layout.bytes));
    }
}

void MaterialTable::ResolveMaterial(uint32_t material) {
    const Prims& prims = m_structure.GetPrims();
    const Paths& paths = m_structure.GetPaths();
    uint32_t surface = kNoPrim;
    bool connected = false;
    for (const Connection& connection : Connections(material)) {
        if (connection.input == "surface") {
            surface = connection.source;
            connected = true;
        }
    }
    if (surface != kNoPrim && prims.Record(surface).schema == SchemaId::UsdPreviewSurface) {
        ResolvePreview(material, surface);
        return;
    }
    if (surface != kNoPrim && IsRuntime(prims.Record(surface).schema)) {
        std::shared_ptr<const material::MaterialNodeDefinition> node =
            material::FindMaterialNode(Info(prims.Record(surface).schema).name);
        if (node && node->role != material::MaterialNodeRole::Pattern) {
            ResolveNode(material, surface, std::move(node));
            return;
        }
    }
    if (!connected) {
        if (GetLogLevel() <= LogLevel::Debug) {
            Log(LogLevel::Debug, "{} has no surface connected, so it shades as plain diffuse",
                paths.Path(material));
        }
    } else {
        Log(LogLevel::Warning,
            "{}'s surface connects to {}, which isn't a surface Ernie knows, so it shades as "
            "plain diffuse",
            paths.Path(material),
            surface == kNoPrim ? std::string("nothing") : paths.Path(surface));
    }
    Define(material, kDefinitionDiffuse, 0);
}

void MaterialTable::Resolve(const ChangeSet& changes) {
    ERNIE_ZONE("Materials/Resolve");
    ++m_resolves;
    const Prims& prims = m_structure.GetPrims();
    m_pack.Clear();
    m_bindings.clear();
    m_spectral.clear();
    m_wordCount = 0;
    m_connections.Update(changes);
    m_touched.assign(prims.Size(), false);
    std::vector<MaterialRecord> before;
    for (uint32_t prim = 0; prim < m_generations.size(); ++prim) {
        before.push_back(m_records.Row(prim));
    }
    m_records.Assign(std::max(prims.Size(), 1u), MaterialRecord{});
    ColumnTable& shaders = m_structure.GetColumnStore().Schema(SchemaId::Shader);
    const std::vector<uint32_t> ids =
        Items<uint32_t>(shaders.Read(kShaderId.index, 0, shaders.Rows()));
    uint32_t materials = 0;
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim)) {
            continue;
        }
        const SchemaId schema = prims.Record(prim).schema;
        if (IsA(schema, SchemaId::Material)) {
            ResolveMaterial(prim);
            ++materials;
        } else if (IsA(schema, SchemaId::Shader)) {
            const std::string_view id =
                m_structure.GetTokens().Text(Token{ids[prims.Row(prim, SchemaId::Shader)]});
            if (id != Info(schema).name) {
                Log(LogLevel::Warning, "{} has info:id {} but is a {}, so it runs {}",
                    m_structure.GetPaths().Path(prim), id, Info(schema).name, Info(schema).name);
            }
        }
    }
    CarryPrevious(before);
    ResolveRadios();
    if (const uint32_t twice = m_pack.Twice(m_wordCount); twice != kNoPrim) {
        Log(LogLevel::Error, "material word {} is packed twice, so its value is undefined", twice);
    }
    Log(LogLevel::Debug, "resolved {} materials into {} words, {} copies and {} textures",
        materials, m_wordCount, m_pack.CopyCount(), m_bindings.size());
}

void MaterialTable::CarryPrevious(const std::vector<MaterialRecord>& before) {
    const Prims& prims = m_structure.GetPrims();
    m_carries.clear();
    std::vector<uint32_t> generations(prims.Size(), 0);
    std::set<uint32_t> used;
    const auto library = m_library.Snapshot();
    const auto definitionWords = [&](uint32_t id) -> uint32_t {
        if (id == kDefinitionPreviewConstant || id == kDefinitionPreviewTextured) {
            return Word(sizeof(PreviewParams));
        }
        const std::shared_ptr<const material::MaterialDefinition> found = library->Find(id);
        return found ? found->words : 0;
    };
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim) || !IsA(prims.Record(prim).schema, SchemaId::Material)) {
            continue;
        }
        generations[prim] = prims.Generation(prim) + 1;
        MaterialRecord record = m_records.Row(prim);
        if (record.definition < material::kFirstGeneratedDefinition) {
            record.previousDefinition = kDefinitionPlaceholder;
            record.previousWords = 0;
            m_records.Set(prim, record);
        } else if (prim < before.size() && m_generations[prim] == generations[prim]) {
            const MaterialRecord& old = before[prim];
            const bool changed = old.definition != record.definition;
            const uint32_t previous = changed ? old.definition : old.previousDefinition;
            const uint32_t from = changed ? old.words : old.previousWords;
            const uint32_t words = definitionWords(previous);
            record.previousDefinition = previous;
            record.previousWords = 0;
            if (words > 0) {
                m_wordCount = (m_wordCount + 1) & ~1u;
                record.previousWords = m_wordCount;
                m_carries.push_back({from, m_wordCount, words});
                m_wordCount += words;
            }
            if (changed && GetLogLevel() <= LogLevel::Debug) {
                Log(LogLevel::Debug,
                    "{} changes from definition {} to {}, and shows {} until the renderer links it",
                    m_structure.GetPaths().Path(prim), old.definition, record.definition,
                    old.definition);
            }
            m_records.Set(prim, record);
        }
        used.insert(record.definition);
        used.insert(record.previousDefinition);
    }
    m_generations = std::move(generations);
    if (!m_carries.empty() && m_words.Buffer()) {
        const Ref<device::Buffer>& arena = m_words.Buffer();
        device::Device& device = m_structure.GetColumnStore().GetDevice();
        m_carried =
            device.CreateBuffer({.bytes = arena->Bytes(), .label = "carried material words"});
        device::Queue& queue = device.GetQueue();
        device::CommandEncoder encoder = queue.CreateEncoder();
        encoder.Copy(arena->Slice(0), m_carried->Slice(0));
        queue.Submit(encoder.Finish());
    } else if (m_carries.empty()) {
        m_carried = nullptr;
    }
    std::vector<material::MaterialDefinition> definitions;
    for (const auto& entry : library->Entries()) {
        if (used.contains(entry.id)) {
            definitions.push_back(*entry.definition);
        }
    }
    auto set = std::make_shared<const material::MaterialDefinitionSet>(
        std::move(definitions), material::ClosurePrimitiveSources(material::ClosurePrimitives()));
    if (set->Hash() != m_definitions->Hash()) {
        Log(LogLevel::Debug, "the scene now uses {}",
            Counted(set->Definitions().size(), "generated material definition"));
        m_definitions = std::move(set);
    }
}

void MaterialTable::ResolveRadios() {
    const Prims& prims = m_structure.GetPrims();
    m_radarLinks.clear();
    for (uint32_t material = 0; material < prims.Size(); ++material) {
        if (!prims.Alive(material) || !IsA(prims.Record(material).schema, SchemaId::Material)) {
            continue;
        }
        for (const Connection& connection : Connections(material)) {
            if (connection.input != "radar") {
                continue;
            }
            if (connection.source == kNoPrim ||
                prims.Record(connection.source).schema != SchemaId::ItuMaterial) {
                Log(LogLevel::Warning,
                    "{}'s radar connects to {}, which isn't an ItuMaterial; radar sees its surface "
                    "instead",
                    m_structure.GetPaths().Path(material),
                    connection.source == kNoPrim ? std::string("nothing")
                                                 : m_structure.GetPaths().Path(connection.source));
                continue;
            }
            m_radarLinks.push_back({material, prims.Row(connection.source, SchemaId::ItuMaterial)});
        }
    }
}

void MaterialTable::WriteRadios() {
    ColumnStore& store = m_structure.GetColumnStore();
    std::vector<RadioRecord> radios(std::max(m_structure.GetPrims().Size(), 1u));
    if (!m_radarLinks.empty()) {
        ColumnTable& itu = store.Schema(SchemaId::ItuMaterial);
        const auto column = [&](ColumnRef ref) { return itu.Read(ref.index, 0, itu.Rows()); };
        const std::vector<uint32_t> kinds = Items<uint32_t>(column(kItuMaterialMaterial));
        const std::vector<float> heights = Items<float>(column(kItuMaterialRmsHeight));
        const std::vector<float> lengths = Items<float>(column(kItuMaterialCorrelationLength));
        for (const RadarLink& link : m_radarLinks) {
            const uint32_t kind = std::min<uint32_t>(kinds[link.row], std::size(kItuMaterials) - 1);
            radios[link.material] = {.kind = 1,
                                     .valid = 1,
                                     .rmsHeight = heights[link.row],
                                     .correlationLength = lengths[link.row],
                                     .data = kItuMaterials[kind]};
        }
    }
    for (uint32_t row = 0; row < radios.size(); ++row) {
        m_radios.Set(row, radios[row]);
    }
    for (uint32_t row = uint32_t(radios.size()); row < m_radios.Rows(); ++row) {
        m_radios.Set(row, RadioRecord{});
    }
    m_radios.Upload(store.GetDevice());
}

void MaterialTable::WarnOutsideVisible() {
    const Prims& prims = m_structure.GetPrims();
    ColumnStore& store = m_structure.GetColumnStore();
    ColumnTable& detectors = store.Schema(SchemaId::DetectorAPI);
    const std::vector<Float2> ranges =
        Items<Float2>(detectors.Read(kDetectorAPIDetectorSpectralRange.index, 0, detectors.Rows()));
    const std::vector<uint32_t> owners = Items<uint32_t>(
        detectors.Read(store.PrimsColumn(SchemaId::DetectorAPI), 0, detectors.Rows()));
    for (uint32_t row = 0; row < ranges.size(); ++row) {
        const uint32_t sensor = owners[row];
        if (sensor == kNoPrim || !prims.Alive(sensor) ||
            (ranges[row].x >= kVisibleFirstNm && ranges[row].y <= kVisibleLastNm)) {
            continue;
        }
        for (uint32_t material = 0; material < prims.Size(); ++material) {
            if (!prims.Alive(material) || !IsA(prims.Record(material).schema, SchemaId::Material) ||
                m_spectral.contains(material) ||
                !m_warned
                     .emplace(uint64_t(prims.Generation(material)) << 32 | material,
                              uint64_t(prims.Generation(sensor)) << 32 | sensor)
                     .second) {
                continue;
            }
            Log(LogLevel::Warning,
                "{} has only RGB colours, but {} sees {} to {} nm, so it sees them extended by a "
                "smooth spectrum; connect a SampledSpectrum for measured values",
                m_structure.GetPaths().Path(material), m_structure.GetPaths().Path(sensor),
                ranges[row].x, ranges[row].y);
        }
    }
}

void MaterialTable::BuildEntries() {
    for (const Binding& binding : m_bindings) {
        const uint64_t handle = m_textures.Handle(binding.texture, "file");
        m_pack.SetLiteral(binding.low, uint32_t(handle));
        m_pack.SetLiteral(binding.high, uint32_t(handle >> 32));
    }
    m_pack.Build(m_structure.GetColumnStore(), true);
}

void MaterialTable::Run(const ChangeSet& changes, TableVersions&) {
    ERNIE_ZONE("Materials/Run");
    ColumnStore& store = m_structure.GetColumnStore();
    device::Device& device = store.GetDevice();
    const uint64_t writes = store.WriteSequence();
    const bool resolve = Restructured(changes) || !m_records.Buffer() ||
                         m_nodeVersion != material::MaterialNodeVersion();
    if (resolve) {
        m_nodeVersion = material::MaterialNodeVersion();
        Resolve(changes);
    }
    const uint32_t prims = m_structure.GetPrims().Size();
    m_records.Reserve(std::max(prims, 1u));
    m_records.Upload(device);
    const bool rebound = resolve || changes.Wrote(TableId::Textures);
    if (rebound) {
        BuildEntries();
    }
    if (resolve) {
        m_words.Reserve(device, std::max(m_wordCount, 1u));
        device::Queue& queue = device.GetQueue();
        device::CommandEncoder encoder = queue.CreateEncoder();
        encoder.Fill(m_words.Buffer()->Slice(0, uint64_t(m_words.Buffer()->Bytes())));
        if (m_carried && !m_carries.empty()) {
            encoder.Barrier();
            for (const Carry& carry : m_carries) {
                encoder.Copy(m_carried->Slice(uint64_t(carry.from) * sizeof(uint32_t),
                                              uint64_t(carry.words) * sizeof(uint32_t)),
                             m_words.Buffer()->Slice(uint64_t(carry.to) * sizeof(uint32_t),
                                                     uint64_t(carry.words) * sizeof(uint32_t)));
            }
        }
        queue.Submit(encoder.Finish());
    }
    if (rebound || m_pack.Stale(store, m_writes)) {
        m_pack.Pack(store, m_words.Address());
    }
    if (resolve || Changed(SchemaId::DetectorAPI)) {
        WarnOutsideVisible();
    }
    m_checkers.Reserve(device, store.Schema(SchemaId::CheckerboardAPI).Capacity());
    m_temperatures.Reserve(device, std::max(prims, 1u));
    m_temperaturePrimvar = m_structure.GetTokens().Intern("ernie:temperature").id;
    if (resolve || prims > m_radios.Rows() || Changed(SchemaId::ItuMaterial)) {
        WriteRadios();
    }
    m_writes = writes;
    if (prims == 0) {
        return;
    }
    MaterialParams params{};
    params.scene.address = store.DeviceData();
    params.checkers.address = m_checkers.Address();
    params.temperatures.address = m_temperatures.Address();
    device::Queue& queue = device.GetQueue();
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Materials/Write");
    device::CommandEncoder encoder = queue.CreateEncoder();
    {
        const device::ZoneScope scope(encoder, zone);
        encoder.Dispatch(Kernels().Entry("WriteMaterials"), params,
                         {.x = device::GroupCount(prims, kMaterialGroup)});
    }
    queue.Submit(encoder.Finish());
}

void MaterialTable::Prepare() {
    Kernels();
}

device::Program& MaterialTable::Kernels() {
    if (!m_program) {
        m_program = m_structure.GetColumnStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.tables.kernels.materials"},
             .library = &kSceneKernels,
             .entries = {"WriteMaterials"},
             .label = "materials"});
    }
    return *m_program;
}

}

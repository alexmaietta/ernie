// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/materials.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <string_view>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/kernels/materials.h"
#include "ernie/scene/tables/textures.h"
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
    std::vector<ColumnRef> inputs = {kPlaneAxis};
    for (const SchemaId schema :
         {SchemaId::Connectable, SchemaId::UsdPreviewSurface, SchemaId::SurfaceDetailAPI,
          SchemaId::UsdUVTexture, SchemaId::UsdPrimvarReader_float2, SchemaId::UsdTransform2d,
          SchemaId::TexturePlacement, SchemaId::NonVisualMaterialAPI, SchemaId::CheckerboardAPI}) {
        const Span<const ColumnInfo> columns = Columns(schema);
        for (uint32_t index = 0; index < columns.size(); ++index) {
            inputs.push_back({schema, &columns[index], index});
        }
    }
    return inputs;
}

bool Restructured(const ChangeSet& changes) {
    if (changes.resync || changes.structure || !changes.added.empty() || !changes.removed.empty() ||
        !changes.apis.empty()) {
        return true;
    }
    for (const ColumnRef column : {kConnectableConnectionsSources, kConnectableConnectionsInputs,
                                   kConnectableConnectionsOutputs}) {
        if (!changes.ArrayRows(column).empty()) {
            return true;
        }
    }
    return false;
}

}

std::span<const ColumnRef> MaterialTable::Inputs() const {
    static const std::vector<ColumnRef> inputs = MaterialInputs();
    return inputs;
}

void MaterialTable::Describe(RenderTables& tables) const {
    tables.materials.address = m_records.Buffer() ? m_records.Buffer()->Address() : 0;
    tables.materialWords.address = m_words.Address();
    tables.nonVisuals.address = m_nonVisuals.Address();
    tables.checkers.address = m_checkers.Address();
}

std::vector<MaterialTable::Connection> MaterialTable::Connections(uint32_t prim) {
    const Prims& prims = m_structure.GetPrims();
    ColumnStore& store = m_structure.GetColumnStore();
    const uint32_t row = prims.Row(prim, SchemaId::Connectable);
    const auto read = [&](ColumnRef column, ColumnType type) {
        const ArrayRef array = store.Array(column, row);
        return array.address == 0 ? std::vector<uint32_t>{}
                                  : Items<uint32_t>(store.Arrays(type).Read(array));
    };
    const std::vector<uint32_t> sources =
        read(kConnectableConnectionsSources, ColumnType::PrimArray);
    const std::vector<uint32_t> inputs =
        read(kConnectableConnectionsInputs, ColumnType::TokenArray);
    const std::vector<uint32_t> outputs =
        read(kConnectableConnectionsOutputs, ColumnType::TokenArray);
    const Tokens& tokens = m_structure.GetTokens();
    std::vector<Connection> connections;
    const size_t count = std::min({sources.size(), inputs.size(), outputs.size()});
    for (size_t i = 0; i < count; ++i) {
        const uint32_t source =
            sources[i] != kNoPrim && prims.Alive(sources[i]) ? sources[i] : kNoPrim;
        connections.push_back({std::string(tokens.Text(Token{inputs[i]})), source,
                               std::string(tokens.Text(Token{outputs[i]}))});
    }
    return connections;
}

void MaterialTable::CopyCell(ColumnRef column, uint32_t prim, uint32_t destination) {
    m_copies.push_back({column, m_structure.GetPrims().Row(prim, column.schema), destination});
}

void MaterialTable::ResolvePlacement(uint32_t placement, uint32_t mapping) {
    const Prims& prims = m_structure.GetPrims();
    const auto at = [&](size_t offset) { return mapping + Word(offset); };
    const auto literal = [&](size_t offset, uint32_t value) {
        m_literals.push_back({at(offset), value});
    };
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

    const std::vector<Connection> connections = Connections(shader);
    uint32_t placement = kNoPrim;
    bool placed = false;
    uint32_t textured = 0;
    for (uint32_t input = 0; input < kPreviewInputs; ++input) {
        const size_t slot = offsetof(PreviewParams, inputs) + input * sizeof(TextureInput);
        const auto unbound = [&] {
            m_literals.push_back({at(slot + offsetof(TextureInput, handle)), 0});
            m_literals.push_back({at(slot + offsetof(TextureInput, handle)) + 1, 0});
        };
        const auto found =
            std::ranges::find(connections, kPreviewInputNames[input], &Connection::input);
        if (found == connections.end()) {
            unbound();
            continue;
        }
        const uint32_t texture = found->source;
        if (texture == kNoPrim || prims.Record(texture).schema != SchemaId::UsdUVTexture) {
            Log(LogLevel::Warning,
                "{}'s {} connects to {}, which isn't a UsdUVTexture; the input keeps its value",
                paths.Path(shader), kPreviewInputNames[input],
                texture == kNoPrim ? std::string("nothing") : paths.Path(texture));
            unbound();
            continue;
        }
        TextureChannel channel = TextureChannel::Rgb;
        if (!Channel(found->output, ColorInput(input), channel)) {
            Log(LogLevel::Warning,
                "{}'s {} reads output {} of {}, which isn't rgb, r, g, b or a; the input keeps "
                "its value",
                paths.Path(shader), kPreviewInputNames[input], found->output, paths.Path(texture));
            unbound();
            continue;
        }
        m_bindings.push_back({texture, at(slot + offsetof(TextureInput, handle))});
        m_literals.push_back({at(slot + offsetof(TextureInput, channel)), uint32_t(channel)});
        CopyCell(kUsdUVTextureScale, texture, at(slot + offsetof(TextureInput, scale)));
        CopyCell(kUsdUVTextureBias, texture, at(slot + offsetof(TextureInput, bias)));
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
    const uint32_t definition =
        textured > 0 ? kDefinitionPreviewTextured : kDefinitionPreviewConstant;
    m_records.Set(material, {.definition = definition, .words = base});
    Log(LogLevel::Debug, "{} shades with {} through {}, with {} textures in {} words",
        paths.Path(material), textured > 0 ? "a textured UsdPreviewSurface" : "UsdPreviewSurface",
        paths.Path(shader), textured, Word(sizeof(PreviewParams)));
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
    if (!connected) {
        Log(LogLevel::Debug, "{} has no surface connected, so it shades as plain diffuse",
            paths.Path(material));
    } else {
        Log(LogLevel::Warning,
            "{}'s surface connects to {}, which isn't a surface Ernie knows, so it shades as "
            "plain diffuse",
            paths.Path(material),
            surface == kNoPrim ? std::string("nothing") : paths.Path(surface));
    }
    m_records.Set(material, {.definition = kDefinitionDiffuse});
}

void MaterialTable::Resolve() {
    ERNIE_ZONE("Materials/Resolve");
    ++m_resolves;
    const Prims& prims = m_structure.GetPrims();
    m_copies.clear();
    m_literals.clear();
    m_bindings.clear();
    m_wordCount = 0;
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
    std::vector<uint32_t> written;
    for (const Copy& copy : m_copies) {
        const uint32_t words = ElementBytes(copy.column.column->type) / sizeof(uint32_t);
        for (uint32_t word = 0; word < words; ++word) {
            written.push_back(copy.destination + word);
        }
    }
    for (const Literal& literal : m_literals) {
        written.push_back(literal.destination);
    }
    for (const Binding& binding : m_bindings) {
        written.push_back(binding.destination);
        written.push_back(binding.destination + 1);
    }
    std::ranges::sort(written);
    if (const auto twice = std::ranges::adjacent_find(written); twice != written.end()) {
        Log(LogLevel::Error, "material word {} is packed twice, so its value is undefined", *twice);
    }
    Log(LogLevel::Debug, "resolved {} materials into {} words, {} copies and {} textures",
        materials, m_wordCount, m_copies.size(), m_bindings.size());
}

void MaterialTable::Pack() {
    ColumnStore& store = m_structure.GetColumnStore();
    device::Device& device = store.GetDevice();
    std::vector<PackEntry> entries;
    entries.reserve(m_copies.size() + m_literals.size() + m_bindings.size());
    for (const Copy& copy : m_copies) {
        ColumnTable& table = store.Schema(copy.column.schema);
        entries.push_back({.pages = table.Pages(copy.column.index),
                           .row = copy.row,
                           .words = table.CellBytes(copy.column.index) / uint32_t(sizeof(uint32_t)),
                           .destination = copy.destination});
    }
    for (const Literal& literal : m_literals) {
        entries.push_back({.words = 1, .destination = literal.destination, .value = literal.value});
    }
    for (const Binding& binding : m_bindings) {
        const uint64_t handle = m_textures.Handle(binding.texture, "file");
        entries.push_back(
            {.words = 1, .destination = binding.destination, .value = uint32_t(handle)});
        entries.push_back(
            {.words = 1, .destination = binding.destination + 1, .value = uint32_t(handle >> 32)});
    }
    m_words.Reserve(device, std::max(m_wordCount, 1u));
    if (entries.empty()) {
        return;
    }
    const uint64_t bytes = entries.size() * sizeof(PackEntry);
    if (!m_entries || m_entries->Bytes() < bytes) {
        m_entries = device.CreateBuffer({.bytes = bytes, .label = "material pack entries"});
    }
    device::Queue& queue = device.GetQueue();
    queue.Write(m_entries->Slice(0, bytes), entries.data());
    PackParams params{};
    params.entries.address = m_entries->Address();
    params.words.address = m_words.Address();
    params.count = uint32_t(entries.size());
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Materials/Pack");
    device::CommandEncoder encoder = queue.CreateEncoder();
    {
        const device::ZoneScope scope(encoder, zone);
        encoder.Dispatch(Kernels().Entry("PackMaterials"), params,
                         {.x = device::GroupCount(params.count, kMaterialGroup)});
    }
    queue.Submit(encoder.Finish());
}

void MaterialTable::Run(const ChangeSet& changes, TableVersions&) {
    ERNIE_ZONE("Materials/Run");
    ColumnStore& store = m_structure.GetColumnStore();
    device::Device& device = store.GetDevice();
    if (Restructured(changes) || !m_records.Buffer()) {
        Resolve();
    }
    m_records.Upload(device);
    Pack();
    m_nonVisuals.Reserve(device, store.Schema(SchemaId::NonVisualMaterialAPI).Capacity());
    m_checkers.Reserve(device, store.Schema(SchemaId::CheckerboardAPI).Capacity());
    const uint32_t prims = m_structure.GetPrims().Size();
    if (prims == 0) {
        return;
    }
    MaterialParams params{};
    params.scene.address = store.DeviceData();
    params.nonVisuals.address = m_nonVisuals.Address();
    params.checkers.address = m_checkers.Address();
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
             .entries = {"WriteMaterials", "PackMaterials"},
             .label = "materials"});
    }
    return *m_program;
}

}

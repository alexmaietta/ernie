// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/materialx.h"

#include "ernie/scene/material_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/value.h"

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Util.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXGenHw/HwConstants.h>
#include <MaterialXGenHw/Nodes/HwSurfaceNode.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/Syntax.h>
#include <MaterialXGenShader/UnitSystem.h>
#include <MaterialXGenShader/Util.h>
#include <MaterialXGenSlang/SlangShaderGenerator.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <variant>

namespace mx = MaterialX;

namespace ernie {
namespace {

constexpr std::string_view kInputs = "inputs:";
constexpr std::string_view kConnect = ".connect";
constexpr std::string_view kOutputs = ".outputs:";

/// MaterialX's data libraries, loaded once.
mx::ConstDocumentPtr Libraries() {
    static std::once_flag once;
    static mx::DocumentPtr libraries;
    std::call_once(once, [] {
        libraries = mx::createDocument();
        const std::filesystem::path folder = MaterialXLibraries();
        const mx::FileSearchPath search(
            folder.parent_path().string());
        mx::loadLibraries({mx::FilePath(folder.filename().string())},
                          search, libraries);
        if (libraries->getNodeDefs().empty()) {
            throw std::runtime_error(
                "MaterialX's data libraries are not in " +
                folder.string() + " (ERNIE_MATERIALX_LIBRARIES)");
        }
    });
    return libraries;
}

/// `value` as MaterialX writes values: numbers, and vectors and
/// matrices as their numbers separated by ", ".
std::string ValueString(const Value& value) {
    std::ostringstream out;
    out << std::setprecision(9);
    const auto numbers = [&](const auto& v, int n) {
        for (int i = 0; i < n; ++i) {
            out << (i == 0 ? "" : ", ") << v[i];
        }
    };
    std::visit(
        [&]<typename T>(const T& v) {
            if constexpr (std::is_same_v<T, bool>) {
                out << (v ? "true" : "false");
            } else if constexpr (std::is_same_v<T, int> ||
                                 std::is_same_v<T, float> ||
                                 std::is_same_v<T, double>) {
                out << v;
            } else if constexpr (std::is_same_v<T, glm::ivec2> ||
                                 std::is_same_v<T, glm::vec2> ||
                                 std::is_same_v<T, glm::dvec2>) {
                numbers(v, 2);
            } else if constexpr (std::is_same_v<T, glm::vec3> ||
                                 std::is_same_v<T, glm::dvec3>) {
                numbers(v, 3);
            } else if constexpr (std::is_same_v<T, glm::vec4> ||
                                 std::is_same_v<T, glm::dvec4>) {
                numbers(v, 4);
            } else if constexpr (std::is_same_v<T, glm::dmat4>) {
                // Row after row, as MaterialX's matrix44 (and glm's
                // columns are USD's rows).
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        out << (r + c == 0 ? "" : ", ") << v[r][c];
                    }
                }
            } else if constexpr (std::is_same_v<T, std::string>) {
                out << v;
            } else if constexpr (std::is_same_v<T, Path>) {
                out << v.String();
            } else if constexpr (std::is_same_v<T,
                                                std::vector<float>>) {
                for (std::size_t i = 0; i < v.size(); ++i) {
                    out << (i == 0 ? "" : ", ") << v[i];
                }
            } else if constexpr (std::is_same_v<
                                     T, std::vector<std::string>>) {
                for (std::size_t i = 0; i < v.size(); ++i) {
                    out << (i == 0 ? "" : ", ") << v[i];
                }
            }
        },
        value.Raw());
    return out.str();
}

/// Builds a MaterialX document from a Material's network of Shader
/// prims: a node per prim, named for its path below the material.
class NetworkWriter {
  public:
    NetworkWriter(const Scene& scene, const Path& material)
        : scene_(scene), material_(material),
          document_(mx::createDocument()) {
        document_->setDataLibrary(Libraries());
    }

    mx::DocumentPtr Document() const {
        return document_;
    }

    /// The node for the Shader prim at `path` and those upstream.
    mx::NodePtr Node(const Path& path) {
        if (const auto it = nodes_.find(path.String());
            it != nodes_.end()) {
            return it->second;
        }
        if (!scene_.Has(path) ||
            scene_.TypeOf(path) != tokens::kShader) {
            throw std::invalid_argument(
                "'" + material_.String() + "' connects to '" +
                path.String() + "', which is not a Shader");
        }
        const std::string id =
            scene_.Get<std::string>(path, tokens::kInfoId, "");
        const mx::NodeDefPtr definition = Libraries()->getNodeDef(id);
        if (!definition) {
            throw std::invalid_argument(
                "'" + path.String() + "' is a '" + id +
                "', which MaterialX does not define");
        }
        const mx::NodePtr node =
            document_->addNodeInstance(definition, NameOf(path));
        nodes_.emplace(path.String(), node);
        for (const std::string& attribute : scene_.Attributes(path)) {
            if (!attribute.starts_with(kInputs)) {
                continue;
            }
            const bool connection = attribute.ends_with(kConnect);
            const std::string name = attribute.substr(
                kInputs.size(),
                attribute.size() - kInputs.size() -
                    (connection ? kConnect.size() : 0));
            const mx::InputPtr input =
                node->addInputFromNodeDef(name);
            if (!input) {
                throw std::invalid_argument(
                    "'" + path.String() + "' (" + id +
                    ") has no input '" + name + "'");
            }
            if (connection) {
                const std::string target =
                    scene_.Get<std::string>(path, attribute, "");
                const auto [upstream, output] = Split(target);
                const mx::NodePtr source = Node(upstream);
                input->setConnectedNode(source);
                if (source->getNodeDef() &&
                    source->getNodeDef()->getOutputCount() > 1) {
                    input->setOutputString(output);
                }
            } else if (const Value* value =
                           scene_.Get(path, attribute)) {
                input->setValueString(ValueString(*value));
            }
        }
        return node;
    }

    /// The prim path and output name of a connection's target,
    /// "<prim>.outputs:<output>".
    static std::pair<Path, std::string>
    Split(const std::string& target) {
        const std::size_t at = target.rfind(kOutputs);
        if (at == std::string::npos) {
            throw std::invalid_argument("'" + target +
                                        "' is not a shader output");
        }
        return {Path(target.substr(0, at)),
                target.substr(at + kOutputs.size())};
    }

  private:
    /// A valid MaterialX name for the prim at `path`: its path below
    /// the material.
    std::string NameOf(const Path& path) const {
        std::string name = path.String();
        const std::string prefix = material_.String() + "/";
        if (name.starts_with(prefix)) {
            name = name.substr(prefix.size());
        }
        return mx::createValidName(name);
    }

    const Scene& scene_;
    Path material_;
    mx::DocumentPtr document_;
    std::unordered_map<std::string, mx::NodePtr> nodes_;
};

void Validate(const mx::DocumentPtr& document,
              const std::string& what) {
    std::string message;
    if (!document->validate(&message)) {
        throw std::invalid_argument(
            what + " is not valid MaterialX: " + message);
    }
}

// ---------------------------------------------------------------------
// Generation for Ernie's shading pass

/// MaterialX's `surface` node with its light loop over Ernie's
/// lights: the closure's response to a unit of each light goes to
/// Ernie (ernieMxLightResponse), which scales it by the light's
/// spectrum, distance and shadow; its environment and emission stay
/// MaterialX's.
///
/// Reference: MaterialX 1.39.5,
/// MaterialXGenHw/Nodes/HwSurfaceNode.cpp (the loop this replaces).
class ErnieSurfaceNode : public mx::HwSurfaceNode {
  public:
    static mx::ShaderNodeImplPtr create() {
        return std::make_shared<ErnieSurfaceNode>();
    }

    void emitLightLoop(const mx::ShaderNode& node,
                       mx::GenContext& context,
                       mx::ShaderStage& stage,
                       const std::string&) const override {
        const auto& generator =
            static_cast<const mx::HwShaderGenerator&>(
                context.getShaderGenerator());
        const mx::ShaderInput* bsdfInput =
            node.getInput(getBsdfInputName());
        const mx::ShaderNode* bsdf =
            bsdfInput != nullptr ? bsdfInput->getConnectedSibling()
                                 : nullptr;
        if (bsdf == nullptr) {
            return;
        }
        generator.emitComment(
            "Ernie's lights: the response to a unit of each, which "
            "Ernie scales by the light",
            stage);
        generator.emitLine("for (int ernieLight = 0; ernieLight < "
                           "ernieMxLightCount(); ++ernieLight)",
                           stage, false);
        generator.emitScopeBegin(stage);
        generator.emitLine("L = ernieMxLightDirection(ernieLight)",
                           stage);
        generator.emitLine("occlusion = 1.0", stage);
        if (bsdf->hasClassification(
                mx::ShaderNode::Classification::BSDF_R)) {
            generator.emitLine(
                "ClosureData closureData = makeClosureData("
                "CLOSURE_TYPE_REFLECTION, L, V, N, P, occlusion)",
                stage);
            generator.emitFunctionCall(*bsdf, context, stage);
        } else {
            generator.emitLineBegin(stage);
            generator.emitOutput(bsdf->getOutput(), true, true,
                                 context, stage);
            generator.emitLineEnd(stage);
        }
        generator.emitLine("ernieMxLightResponse(ernieLight, " +
                               bsdf->getOutput()->getVariable() +
                               ".response)",
                           stage);
        generator.emitScopeEnd(stage);
        generator.emitLineBreak(stage);
    }
};

/// The bytes a uniform of MaterialX type `type` takes in the uniform
/// buffer (scalar layout), or 0 for a type it does not hold.
std::uint32_t UniformBytes(const std::string& type) {
    static const std::unordered_map<std::string, std::uint32_t>
        kBytes{{"float", 4},    {"integer", 4},  {"boolean", 4},
               {"vector2", 8},  {"vector3", 12}, {"color3", 12},
               {"vector4", 16}, {"color4", 16},  {"matrix33", 36},
               {"matrix44", 64}};
    const auto it = kBytes.find(type);
    return it != kBytes.end() ? it->second : 0;
}

/// The bridge call reading a uniform of MaterialX type `type` at byte
/// `offset`.
std::string UniformLoad(const std::string& type,
                        std::uint32_t offset) {
    static const std::unordered_map<std::string, std::string> kLoads{
        {"float", "ernieMxLoadFloat"},
        {"integer", "ernieMxLoadInt"},
        {"vector2", "ernieMxLoadFloat2"},
        {"vector3", "ernieMxLoadFloat3"},
        {"color3", "ernieMxLoadFloat3"},
        {"vector4", "ernieMxLoadFloat4"},
        {"color4", "ernieMxLoadFloat4"},
        {"matrix33", "ernieMxLoadFloat3x3"},
        {"matrix44", "ernieMxLoadFloat4x4"}};
    const std::string at = std::to_string(offset) + "u";
    if (type == "boolean") {
        return "(ernieMxLoadInt(" + at + ") != 0)";
    }
    return kLoads.at(type) + "(" + at + ")";
}

/// What Ernie gives for MaterialX's private uniforms it knows (the
/// view's position, the time, the transforms), by variable; empty for
/// one left at MaterialX's default.
std::string PrivateUniform(const std::string& variable) {
    static const std::unordered_map<std::string, std::string> kBridge{
        {"u_viewPosition", "ernieMxViewPosition()"},
        {"u_time", "ernieMxTime()"},
        {"u_frame", "ernieMxFrame()"},
        {"u_worldMatrix", "ernieMxWorldMatrix()"},
        {"u_worldInverseMatrix", "ernieMxWorldInverseMatrix()"},
        {"u_worldTransposeMatrix", "transpose(ernieMxWorldMatrix())"},
        {"u_worldInverseTransposeMatrix",
         "transpose(ernieMxWorldInverseMatrix())"},
        {"u_viewMatrix", "ernieMxViewMatrix()"},
        {"u_viewInverseMatrix", "ernieMxViewInverseMatrix()"},
        {"u_projectionMatrix", "ernieMxProjectionMatrix()"}};
    const auto it = kBridge.find(variable);
    return it != kBridge.end() ? it->second : std::string();
}

/// What fills a vertex data field (`VertexData`'s, as MaterialX's
/// pixel stages read them) of variable `variable` and Slang type
/// `type`, from Ernie; its primvar's name to `geomProps` when it is
/// one.
std::string VertexDataFill(const std::string& variable,
                           const std::string& type,
                           std::vector<std::string>& geomProps) {
    static const std::unordered_map<std::string, std::string> kFields{
        {"positionWorld", "ernieMxPositionWorld()"},
        {"normalWorld", "ernieMxNormalWorld()"},
        {"tangentWorld", "ernieMxTangentWorld()"},
        {"bitangentWorld", "ernieMxBitangentWorld()"},
        {"positionObject", "ernieMxPositionObject()"},
        {"normalObject", "ernieMxNormalObject()"},
        {"tangentObject", "ernieMxTangentObject()"},
        {"bitangentObject", "ernieMxBitangentObject()"}};
    if (const auto it = kFields.find(variable); it != kFields.end()) {
        return it->second;
    }
    const auto set = [&](std::string_view prefix) {
        return std::stoi(variable.substr(prefix.size()));
    };
    if (variable.starts_with("texcoord_")) {
        const std::string uv = "ernieMxTexcoord(" +
                               std::to_string(set("texcoord_")) + ")";
        return type == "float3" ? "float3(" + uv + ", 0.0)" : uv;
    }
    if (variable.starts_with("color_")) {
        const std::string colour =
            "ernieMxColor(" + std::to_string(set("color_")) + ")";
        return type == "float3" ? colour + ".rgb" : colour;
    }
    if (variable.starts_with("geomprop_")) {
        const auto id = static_cast<int>(geomProps.size());
        geomProps.push_back(
            variable.substr(std::string("geomprop_").size()));
        static const std::unordered_map<std::string, std::string>
            kTypes{{"float", "Float"},
                   {"float2", "Float2"},
                   {"float3", "Float3"},
                   {"float4", "Float4"},
                   {"int", "Int"}};
        const auto it = kTypes.find(type);
        if (it != kTypes.end()) {
            return "ernieMxGeomProp" + it->second + "(" +
                   std::to_string(id) + ")";
        }
    }
    return {};
}

/// The bridge functions MaterialX shaders call (`MaterialXShader`),
/// declared extern: the shading pass that imports the generated
/// module defines them (`export`), and Slang links the two.
constexpr const char* kBridgeDeclarations =
    R"(// Ernie's bridge (ernie/session/materialx.h, MaterialXShader).
extern float3 ernieMxPositionWorld();
extern float3 ernieMxNormalWorld();
extern float3 ernieMxTangentWorld();
extern float3 ernieMxBitangentWorld();
extern float3 ernieMxPositionObject();
extern float3 ernieMxNormalObject();
extern float3 ernieMxTangentObject();
extern float3 ernieMxBitangentObject();
extern float2 ernieMxTexcoord(int set);
extern float4 ernieMxColor(int set);
extern float ernieMxGeomPropFloat(int id);
extern float2 ernieMxGeomPropFloat2(int id);
extern float3 ernieMxGeomPropFloat3(int id);
extern float4 ernieMxGeomPropFloat4(int id);
extern int ernieMxGeomPropInt(int id);
extern float3 ernieMxViewPosition();
extern float ernieMxTime();
extern float ernieMxFrame();
extern float4x4 ernieMxWorldMatrix();
extern float4x4 ernieMxWorldInverseMatrix();
extern float4x4 ernieMxViewMatrix();
extern float4x4 ernieMxViewInverseMatrix();
extern float4x4 ernieMxProjectionMatrix();
extern float ernieMxLoadFloat(uint offset);
extern int ernieMxLoadInt(uint offset);
extern float2 ernieMxLoadFloat2(uint offset);
extern float3 ernieMxLoadFloat3(uint offset);
extern float4 ernieMxLoadFloat4(uint offset);
extern float3x3 ernieMxLoadFloat3x3(uint offset);
extern float4x4 ernieMxLoadFloat4x4(uint offset);
extern float4 ernieMxSample(uint slot, float2 uv);
extern float4 ernieMxSampleLevel(uint slot, float2 uv, float lod);
extern float4 ernieMxSampleGrad(uint slot, float2 uv, float2 dx, float2 dy);
extern int2 ernieMxTextureSize(uint slot, int level);
extern int ernieMxTextureLevels(uint slot);
extern int ernieMxLightCount();
extern float3 ernieMxLightDirection(int light);
extern void ernieMxLightResponse(int light, float3 response);
extern float3 ernieMxEnvironmentRadiance(float3 direction, float alpha);
extern float3 ernieMxEnvironmentIrradiance(float3 normal);

)";

/// Corrects what MaterialX 1.39.5's Slang generation leaves invalid
/// in `code`, where it turns its shared GLSL into Slang token by
/// token (SlangShaderGenerator::SlangSyntaxFromGlsl):
/// - GLSL's `mix`, renamed `lerp` where it stands alone but not where
///   punctuation follows it, as a parameter of that name is used
///   (LamaMix's "mix" input);
/// - a global `const` without `static`, which Slang requires, where
/// its
///   own fix runs before `mat3` becomes `float3x3`
///   (mx_blackbody.glsl);
/// - GLSL's matrix times vector, which Slang spells `mul`
///   (mx_flake.glsl).
void FixMaterialXSlang(std::string& code) {
    // A use of `mix` that is not a call: the variable it named.
    static const std::regex kMix(R"(\bmix\b(\s*[^\s(a-zA-Z0-9_]))");
    code = std::regex_replace(code, kMix, "lerp$1");
    static const std::regex kGlobalConst(
        R"((^|\n)const (float3x3|mat3) XYZ_to_RGB)");
    code = std::regex_replace(code, kGlobalConst,
                              "$1static const float3x3 XYZ_to_RGB");
    const std::string flake = "return m * p;";
    for (std::size_t at = code.find(flake); at != std::string::npos;
         at = code.find(flake, at)) {
        code.replace(at, flake.size(), "return mx_matrix_mul(m, p);");
    }
}

/// MaterialX's Slang generator, writing a function Ernie's shading
/// pass calls (`MaterialXShader`) in place of a fragment shader: its
/// textures, uniforms, vertex data, lights and environment through
/// Ernie's bridge functions.
///
/// References:
/// - MaterialX 1.39.5, MaterialXGenSlang/SlangShaderGenerator.cpp
/// (the
///   pixel stage this adapts).
/// - Pixar, OpenUSD hdSt/materialXShaderGen.cpp (a renderer's
/// subclass
///   of MaterialX's generator onto its own lights and data).
class ErnieSlangShaderGenerator : public mx::SlangShaderGenerator {
  public:
    explicit ErnieSlangShaderGenerator(MaterialXShader& out)
        : SlangShaderGenerator(mx::TypeSystem::create()), out_(out) {
        registerImplementation("IM_surface_" + TARGET,
                               ErnieSurfaceNode::create);
    }

  protected:
    void emitPixelStage(const mx::ShaderGraph& graph,
                        mx::GenContext& context,
                        mx::ShaderStage& stage) const override {
        emitDirectives(context, stage);
        emitString(kBridgeDeclarations, stage);
        emitTextures(stage);
        emitTypeDefinitions(context, stage);
        emitConstants(context, stage);
        const bool lighting = requiresLighting(graph);
        emitUniforms(context, stage, false);
        emitInputs(context, stage);
        emitSetup(context, stage);
        emitLibraryInclude("stdlib/genslang/lib/mx_math.slang",
                           context, stage);
        emitLineBreak(stage);
        emitLine(
            "#define DIRECTIONAL_ALBEDO_METHOD " +
                std::to_string(int(
                    context.getOptions().hwDirectionalAlbedoMethod)),
            stage, false);
        emitLine(
            "#define AIRY_FRESNEL_ITERATIONS " +
                std::to_string(
                    context.getOptions().hwAiryFresnelIterations),
            stage, false);
        emitLineBreak(stage);
        if (lighting) {
            emitSpecularEnvironment(context, stage);
            emitTransmissionRender(context, stage);
        }
        _tokenSubstitutions
            [mx::ShaderGenerator::T_FILE_TRANSFORM_UV] =
                context.getOptions().fileTextureVerticalFlip
                    ? "mx_transform_uv_vflip.glsl"
                    : "mx_transform_uv.glsl";
        _tokenSubstitutions[mx::HW::T_TEX_SAMPLER_SIGNATURE] =
            "SamplerTexture2D tex_sampler";
        emitFunctionDefinitions(graph, context, stage);

        const mx::ShaderGraphOutputSocket* socket =
            graph.getOutputSocket();
        setFunctionName("ernieMaterialX", stage);
        emitLine(
            "void ernieMaterialX(out float3 ernieColor, out float3 "
            "ernieTransparency)",
            stage, false);
        emitFunctionBodyBegin(graph, context, stage);
        emitLine("ernieMxSetup()", stage);
        emitLine("ernieColor = float3(0.0)", stage);
        emitLine("ernieTransparency = float3(0.0)", stage);
        const bool surface = graph.hasClassification(
            mx::ShaderNode::Classification::SHADER |
            mx::ShaderNode::Classification::SURFACE);
        out_.surface = graph.hasClassification(
            mx::ShaderNode::Classification::SURFACE);
        if (graph.hasClassification(
                mx::ShaderNode::Classification::CLOSURE) &&
            !graph.hasClassification(
                mx::ShaderNode::Classification::SHADER)) {
            // A bare closure, unattached to a surface: black, as
            // MaterialX's own renderers show it.
        } else if (surface) {
            emitFunctionCalls(
                graph, context, stage,
                mx::ShaderNode::Classification::TEXTURE);
            for (mx::ShaderGraphOutputSocket* output :
                 graph.getOutputSockets()) {
                if (output->getConnection() == nullptr) {
                    continue;
                }
                const mx::ShaderNode* upstream =
                    output->getConnection()->getNode();
                if (upstream->getParent() == &graph &&
                    (upstream->hasClassification(
                         mx::ShaderNode::Classification::CLOSURE) ||
                     upstream->hasClassification(
                         mx::ShaderNode::Classification::SHADER))) {
                    emitFunctionCall(*upstream, context, stage);
                }
            }
        } else {
            emitFunctionCalls(graph, context, stage);
        }
        if (const mx::ShaderOutput* connection =
                socket->getConnection()) {
            const std::string variable = connection->getVariable();
            const std::string type = connection->getType().getName();
            if (out_.surface) {
                emitLine("ernieColor = " + variable + ".color",
                         stage);
                emitLine("ernieTransparency = " + variable +
                             ".transparency",
                         stage);
            } else if (type == "float" || type == "integer" ||
                       type == "boolean") {
                emitLine("ernieColor = float3(float(" + variable +
                             "))",
                         stage);
            } else if (type == "vector2") {
                emitLine("ernieColor = float3(" + variable + ", 0.0)",
                         stage);
            } else if (type == "vector3" || type == "color3") {
                emitLine("ernieColor = " + variable, stage);
            } else if (type == "vector4" || type == "color4") {
                emitLine("ernieColor = " + variable + ".rgb", stage);
            }
        }
        emitFunctionBodyEnd(graph, context, stage);
    }

    /// Uniforms as static variables `ernieMxSetup` fills: textures as
    /// slots of the texture table, the values Ernie knows (the view's
    /// position, the time) from Ernie, the rest from the material's
    /// uniform buffer.
    void emitUniforms(mx::GenContext& context, mx::ShaderStage& stage,
                      bool) const override {
        loads_.clear();
        std::uint32_t offset = 0;
        for (const auto& [name, block] : stage.getUniformBlocks()) {
            if (block->getName() == mx::HW::LIGHT_DATA ||
                block->empty()) {
                continue;
            }
            emitComment("Uniform block: " + block->getName(), stage);
            for (std::size_t i = 0; i < block->size(); ++i) {
                const mx::ShaderPort* port = (*block)[i];
                const std::string& variable = port->getVariable();
                const std::string type = port->getType().getName();
                const std::string value =
                    port->getValue()
                        ? port->getValue()->getValueString()
                        : std::string();
                if (type == "filename" &&
                    block->getName() == mx::HW::PRIVATE_UNIFORMS) {
                    // MaterialX's environment maps and albedo table,
                    // which Ernie's environment replaces.
                    emitLine("static const SamplerTexture2D " +
                                 variable + " = { 0xFFFFFFFFu }",
                             stage);
                    continue;
                }
                if (type == "filename") {
                    MaterialXTexture texture;
                    texture.variable = variable;
                    texture.slot = static_cast<std::uint32_t>(
                        out_.textures.size());
                    texture.file = value;
                    texture.path = port->getPath();
                    out_.textures.push_back(texture);
                    emitLine("static const SamplerTexture2D " +
                                 variable + " = { " +
                                 std::to_string(texture.slot) + "u }",
                             stage);
                    continue;
                }
                const std::string bridge = PrivateUniform(variable);
                const std::uint32_t bytes = UniformBytes(type);
                if (bridge.empty() &&
                    (bytes == 0 ||
                     block->getName() == mx::HW::PRIVATE_UNIFORMS)) {
                    // Strings and MaterialX's own renderer's
                    // settings: their values as they are.
                    emitLineBegin(stage);
                    emitVariableDeclaration(port, "static", context,
                                            stage, true);
                    emitString(mx::Syntax::SEMICOLON, stage);
                    emitLineEnd(stage, false);
                    continue;
                }
                emitLineBegin(stage);
                emitVariableDeclaration(port, "static", context,
                                        stage, false);
                emitString(mx::Syntax::SEMICOLON, stage);
                emitLineEnd(stage, false);
                if (!bridge.empty()) {
                    loads_.push_back(variable + " = " + bridge);
                    continue;
                }
                loads_.push_back(variable + " = " +
                                 UniformLoad(type, offset));
                out_.uniforms.push_back(
                    {variable, type, offset, port->getPath(), value});
                offset += bytes;
            }
            emitLineBreak(stage);
        }
        out_.uniformBytes = offset;
    }

    /// The light from all around as Ernie gives it, in place of
    /// MaterialX's environment maps: the same lobes (GGX about the
    /// mirror or refracted direction, with its directional albedo)
    /// and irradiance.
    ///
    /// Reference: MaterialX 1.39.5,
    /// pbrlib/genglsl/lib/mx_environment_prefilter.glsl (the
    /// functions these replace).
    void
    emitSpecularEnvironment(mx::GenContext& context,
                            mx::ShaderStage& stage) const override {
        emitLibraryInclude(
            "pbrlib/genglsl/lib/mx_microfacet_specular.glsl", context,
            stage);
        emitString(R"(
float3 mx_environment_radiance(float3 N, float3 V, float3 X, float2 alpha, int distribution, FresnelData fd)
{
    N = mx_forward_facing_normal(N, V);
    float3 L = fd.refraction ? mx_refraction_solid_sphere(-V, N, fd.ior.x) : -reflect(V, N);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(alpha);
    float3 FG = mx_ggx_dir_albedo(NdotV, avgAlpha, fd);
    FG = fd.refraction ? float3(1.0) - FG : FG;
    return ernieMxEnvironmentRadiance(L, avgAlpha) * FG;
}

float3 mx_environment_irradiance(float3 N)
{
    return ernieMxEnvironmentIrradiance(N);
}

)",
                   stage);
    }

    /// No MaterialX light functions: the light loop is Ernie's.
    void
    emitLightFunctionDefinitions(const mx::ShaderGraph&,
                                 mx::GenContext&,
                                 mx::ShaderStage&) const override {}

  private:
    /// Textures as slots of the material's texture table, sampled
    /// through Ernie, in place of MaterialX's mx_texture.slang.
    void emitTextures(mx::ShaderStage& stage) const {
        emitString(R"(struct SamplerTexture2D
{
    uint slot;
    int get_width(int mipLevel = 0) { return ernieMxTextureSize(slot, mipLevel).x; }
    int get_height(int mipLevel = 0) { return ernieMxTextureSize(slot, mipLevel).y; }
    int get_num_mip_levels() { return ernieMxTextureLevels(slot); }
}

float4 textureLod(SamplerTexture2D tex, float2 uv, float lod) { return ernieMxSampleLevel(tex.slot, uv, lod); }
float4 texture(SamplerTexture2D tex, float2 uv) { return ernieMxSample(tex.slot, uv); }
float4 textureGrad(SamplerTexture2D tex, float2 uv, float2 ddx_, float2 ddy_) { return ernieMxSampleGrad(tex.slot, uv, ddx_, ddy_); }
int2 textureSize(SamplerTexture2D tex, int mipLevel) { return ernieMxTextureSize(tex.slot, mipLevel); }

)",
                   stage);
    }

    /// `ernieMxSetup`: the uniforms' values and the vertex data's.
    void emitSetup(mx::GenContext&, mx::ShaderStage& stage) const {
        emitLine("void ernieMxSetup()", stage, false);
        emitScopeBegin(stage);
        for (const std::string& load : loads_) {
            emitLine(load, stage);
        }
        const mx::VariableBlock& vertexData =
            stage.getInputBlock(mx::HW::VERTEX_DATA);
        for (std::size_t i = 0; i < vertexData.size(); ++i) {
            const mx::ShaderPort* port = vertexData[i];
            const std::string fill =
                VertexDataFill(port->getVariable(),
                               _syntax->getTypeName(port->getType()),
                               out_.geomProps);
            if (!fill.empty()) {
                emitLine("vd." + port->getVariable() + " = " + fill,
                         stage);
            }
        }
        emitScopeEnd(stage);
        emitLineBreak(stage);
    }

    MaterialXShader& out_;
    mutable std::vector<std::string> loads_;
};

}  // namespace

std::vector<std::string>
MaterialXElements(const std::filesystem::path& file) {
    const mx::DocumentPtr document = mx::createDocument();
    try {
        mx::readFromXmlFile(document, file.string());
    } catch (const std::exception& e) {
        throw std::invalid_argument("Cannot read " + file.string() +
                                    ": " + e.what());
    }
    document->setDataLibrary(Libraries());
    std::vector<std::string> elements;
    for (const mx::TypedElementPtr& element :
         mx::findRenderableElements(document)) {
        elements.push_back(element->getNamePath());
    }
    return elements;
}

std::filesystem::path MaterialXLibraries() {
    if (const char* folder = std::getenv("ERNIE_MATERIALX_LIBRARIES");
        folder != nullptr && *folder != '\0') {
        return folder;
    }
#ifdef ERNIE_MATERIALX_LIBRARIES_BUILD_PATH
    return ERNIE_MATERIALX_LIBRARIES_BUILD_PATH;
#else
    return {};
#endif
}

std::optional<MaterialXSource> MaterialXOf(const Scene& scene,
                                           const Path& material) {
    if (!scene.Has(material) ||
        scene.TypeOf(material) != tokens::kMaterial) {
        return std::nullopt;
    }
    const std::string surface = scene.Get<std::string>(
        material, std::string(tokens::kMtlxSurface) + ".connect", "");
    if (!surface.empty()) {
        NetworkWriter writer(scene, material);
        const auto [shader, output] = NetworkWriter::Split(surface);
        const mx::NodePtr node = writer.Node(shader);
        const mx::NodePtr surfaceMaterial =
            writer.Document()->addMaterialNode(
                mx::createValidName(std::string(material.Name())),
                node);
        Validate(writer.Document(), "'" + material.String() + "'");
        return MaterialXSource{
            mx::writeToXmlString(writer.Document()),
            surfaceMaterial->getNamePath(),
            {}};
    }
    const Material view(scene, material);
    const std::string file = view.GetMaterialXFile();
    if (file.empty()) {
        return std::nullopt;
    }
    const std::filesystem::path path(file);
    const mx::DocumentPtr document = mx::createDocument();
    try {
        mx::readFromXmlFile(document, path.string());
    } catch (const std::exception& e) {
        throw std::invalid_argument("'" + material.String() +
                                    "' cannot read " + file + ": " +
                                    e.what());
    }
    document->setDataLibrary(Libraries());
    Validate(document, file);
    std::string element = view.GetMaterialXElement();
    if (element.empty()) {
        const std::vector<mx::TypedElementPtr> renderable =
            mx::findRenderableElements(document);
        if (renderable.empty()) {
            throw std::invalid_argument(file +
                                        " has nothing to render");
        }
        element = renderable.front()->getNamePath();
    } else if (!document->getDescendant(element)) {
        throw std::invalid_argument(file + " has no element '" +
                                    element + "'");
    }
    return MaterialXSource{mx::writeToXmlString(document), element,
                           path.parent_path()};
}

MaterialXShader
GenerateMaterialXShader(const MaterialXSource& source) {
    const mx::DocumentPtr document = mx::createDocument();
    mx::FileSearchPath search;
    if (!source.base.empty()) {
        search.append(mx::FilePath(source.base.string()));
    }
    mx::readFromXmlString(document, source.document, search);
    const mx::ConstDocumentPtr libraries = Libraries();
    document->setDataLibrary(libraries);
    const mx::ElementPtr found =
        document->getDescendant(source.element);
    const mx::TypedElementPtr element =
        found ? found->asA<mx::TypedElement>() : nullptr;
    if (!element) {
        throw std::invalid_argument("The MaterialX document has no "
                                    "element '" +
                                    source.element + "' to render");
    }

    MaterialXShader out;
    const auto generator =
        std::make_shared<ErnieSlangShaderGenerator>(out);
    // Colour spaces (textures' sRGB and the like) and units (lengths,
    // angles) as the standard libraries define them.
    const mx::DefaultColorManagementSystemPtr colours =
        mx::DefaultColorManagementSystem::create(
            generator->getTarget());
    colours->loadLibrary(
        std::const_pointer_cast<mx::Document>(libraries));
    generator->setColorManagementSystem(colours);
    const mx::UnitSystemPtr units =
        mx::UnitSystem::create(generator->getTarget());
    units->loadLibrary(
        std::const_pointer_cast<mx::Document>(libraries));
    const mx::UnitConverterRegistryPtr converters =
        mx::UnitConverterRegistry::create();
    for (const char* kind : {"distance", "angle"}) {
        if (const mx::UnitTypeDefPtr type =
                libraries->getUnitTypeDef(kind)) {
            converters->addUnitConverter(
                type, mx::LinearUnitConverter::create(type));
        }
    }
    units->setUnitConverterRegistry(converters);
    generator->setUnitSystem(units);

    // The document's own struct types (MaterialX 1.39's typedefs).
    generator->registerTypeDefs(document);
    mx::GenContext context(generator);
    const std::filesystem::path folder = MaterialXLibraries();
    context.registerSourceCodeSearchPath(
        mx::FilePath(folder.parent_path().string()));
    context.registerSourceCodeSearchPath(
        mx::FilePath(folder.string()));
    mx::GenOptions& options = context.getOptions();
    options.targetColorSpaceOverride = "lin_rec709";
    options.targetDistanceUnit = "meter";
    // Ernie's images keep their top row first, where texture
    // coordinates have their origin at the bottom, as USD's do.
    options.fileTextureVerticalFlip = true;
    options.hwTransparency = true;
    options.hwShadowMap = false;
    options.hwAmbientOcclusion = false;
    try {
        out.code = generator
                       ->generate(mx::createValidName(source.element),
                                  element, context)
                       ->getSourceCode(mx::Stage::PIXEL);
        FixMaterialXSlang(out.code);
    } catch (const std::exception& e) {
        throw std::invalid_argument("MaterialX cannot generate '" +
                                    source.element +
                                    "': " + e.what());
    }
    out.transparent =
        mx::isTransparentSurface(element, generator->getTarget());
    // Each texture's file relative to its document, and its image
    // node's addressing and filtering, from its uniforms beside it
    // ("<node>_file", "<node>_uaddressmode" and so on).
    for (MaterialXTexture& texture : out.textures) {
        if (!texture.file.empty() && !source.base.empty() &&
            std::filesystem::path(texture.file).is_relative()) {
            texture.file = (source.base / texture.file).string();
        }
        const std::size_t at = texture.variable.rfind("file");
        const std::string node = texture.variable.substr(0, at);
        for (const MaterialXUniform& uniform : out.uniforms) {
            if (uniform.value.empty() ||
                !uniform.variable.starts_with(node)) {
                continue;
            }
            const std::string name =
                uniform.variable.substr(node.size());
            if (name == "uaddressmode") {
                texture.uAddress = std::stoi(uniform.value);
            } else if (name == "vaddressmode") {
                texture.vAddress = std::stoi(uniform.value);
            } else if (name == "filtertype") {
                texture.filter = std::stoi(uniform.value);
            }
        }
    }
    return out;
}

}  // namespace ernie

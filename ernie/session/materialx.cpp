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
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/UnitSystem.h>
#include <MaterialXGenShader/Util.h>
#include <MaterialXGenSlang/SlangShaderGenerator.h>

#include <cstdlib>
#include <iomanip>
#include <mutex>
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

}  // namespace

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

MaterialXSlang GenerateMaterialXSlang(const MaterialXSource& source) {
    const mx::DocumentPtr document = mx::createDocument();
    mx::FileSearchPath search;
    if (!source.base.empty()) {
        search.append(mx::FilePath(source.base.string()));
    }
    mx::readFromXmlString(document, source.document, search);
    const mx::ConstDocumentPtr libraries = Libraries();
    document->setDataLibrary(libraries);
    const mx::TypedElementPtr element =
        document->getDescendant(source.element)
            ? document->getDescendant(source.element)
                  ->asA<mx::TypedElement>()
            : nullptr;
    if (!element) {
        throw std::invalid_argument("The MaterialX document has no "
                                    "element '" +
                                    source.element + "' to render");
    }

    const mx::ShaderGeneratorPtr generator =
        mx::SlangShaderGenerator::create();
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

    mx::GenContext context(generator);
    const std::filesystem::path folder = MaterialXLibraries();
    context.registerSourceCodeSearchPath(
        mx::FilePath(folder.parent_path().string()));
    context.registerSourceCodeSearchPath(
        mx::FilePath(folder.string()));
    context.getOptions().targetColorSpaceOverride = "lin_rec709";
    context.getOptions().targetDistanceUnit = "meter";
    mx::ShaderPtr shader;
    try {
        shader = generator->generate(
            mx::createValidName(source.element), element, context);
    } catch (const std::exception& e) {
        throw std::invalid_argument("MaterialX cannot generate '" +
                                    source.element +
                                    "': " + e.what());
    }
    return MaterialXSlang{shader->getSourceCode(mx::Stage::VERTEX),
                          shader->getSourceCode(mx::Stage::PIXEL)};
}

}  // namespace ernie

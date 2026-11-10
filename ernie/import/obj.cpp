// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/import/obj.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include <fmt/format.h>
#include <tiny_obj_loader.h>

#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"
#include "ernie/util/text.h"

namespace ernie {
namespace {

std::string ToIdentifier(std::string_view text) {
    std::string identifier;
    for (const unsigned char c : text) {
        identifier += std::isalnum(c) ? char(c) : '_';
    }
    if (!identifier.empty() && std::isdigit((unsigned char)identifier.front())) {
        identifier.insert(0, "_");
    }
    return identifier;
}

class ChildNames {
  public:
    std::string Take(std::string_view name, std::string_view fallback) {
        std::string identifier = ToIdentifier(name);
        if (!IsIdentifier(identifier)) {
            identifier = std::string(fallback);
        }
        std::string taken = identifier;
        for (int n = 2; !m_taken.insert(taken).second; ++n) {
            taken = fmt::format("{}_{}", identifier, n);
        }
        return taken;
    }

  private:
    std::set<std::string> m_taken;
};

Float3 Cross(Float3 a, Float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Float3 Normalize(Float3 v) {
    const float length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return length > 0 ? Float3{v.x / length, v.y / length, v.z / length} : Float3{0, 0, 1};
}

class MeshBuilder {
  public:
    MeshBuilder(const tinyobj::attrib_t& attributes, const std::filesystem::path& file)
        : m_attributes(attributes), m_file(file),
          m_colors(!attributes.colors.empty() &&
                   attributes.colors.size() == attributes.vertices.size()) {}

    void AddFace(std::span<const tinyobj::index_t> corners, uint32_t smoothing) {
        m_desc.faceVertexCounts.push_back(int32_t(corners.size()));
        m_smoothing.push_back(smoothing);
        for (const tinyobj::index_t& corner : corners) {
            m_desc.faceVertexIndices.push_back(Point(corner.vertex_index));
            AddNormal(corner.normal_index);
            AddSt(corner.texcoord_index);
        }
    }

    Mesh::Desc Finish(std::string material) && {
        if (!m_normals) {
            m_desc.normals = SmoothNormals();
        }
        m_desc.normalsInterpolation = Interpolation::FaceVarying;
        if (m_st) {
            m_desc.primvars.push_back(Primvar::Create("st", Interpolation::FaceVarying, m_stValues,
                                                      std::move(m_stIndices)));
        }
        if (m_colors) {
            m_desc.primvars.push_back(
                Primvar::Create("displayColor", Interpolation::Vertex, m_colorValues));
        }
        if (!material.empty()) {
            m_desc.materialBinding = {std::move(material)};
        }
        m_desc.subdivisionScheme = SubdivisionScheme::None;
        return std::move(m_desc);
    }

    uint32_t FaceCount() const {
        return uint32_t(m_desc.faceVertexCounts.size());
    }

    uint32_t PointCount() const {
        return uint32_t(m_desc.points.size());
    }

  private:
    void Check(int index, size_t count, std::string_view what) const {
        if (index < 0 || size_t(index) >= count) {
            throw Error("Reading \"{}\" as OBJ: a face uses {} {} of {}", m_file.string(), what,
                        index + 1, count);
        }
    }

    int32_t Point(int vertex) {
        const std::vector<tinyobj::real_t>& v = m_attributes.vertices;
        Check(vertex, v.size() / 3, "vertex");
        const auto [found, added] = m_points.try_emplace(vertex, int32_t(m_desc.points.size()));
        if (added) {
            m_desc.points.push_back({v[3 * vertex], v[3 * vertex + 1], v[3 * vertex + 2]});
            if (m_colors) {
                const std::vector<tinyobj::real_t>& c = m_attributes.colors;
                m_colorValues.push_back({c[3 * vertex], c[3 * vertex + 1], c[3 * vertex + 2]});
            }
        }
        return found->second;
    }

    void AddNormal(int normal) {
        if (normal < 0) {
            m_normals = false;
        }
        if (!m_normals) {
            return;
        }
        const std::vector<tinyobj::real_t>& n = m_attributes.normals;
        Check(normal, n.size() / 3, "normal");
        m_desc.normals.push_back({n[3 * normal], n[3 * normal + 1], n[3 * normal + 2]});
    }

    void AddSt(int texcoord) {
        if (texcoord < 0) {
            m_st = false;
        }
        if (!m_st) {
            return;
        }
        const std::vector<tinyobj::real_t>& t = m_attributes.texcoords;
        Check(texcoord, t.size() / 2, "texture coordinate");
        const auto [found, added] = m_stSlots.try_emplace(texcoord, int32_t(m_stValues.size()));
        if (added) {
            m_stValues.push_back({t[2 * texcoord], t[2 * texcoord + 1]});
        }
        m_stIndices.push_back(found->second);
    }

    std::vector<Float3> SmoothNormals() const {
        if (std::ranges::all_of(m_smoothing, [](uint32_t group) { return group == 0; })) {
            return {};
        }
        const std::vector<Float3>& points = m_desc.points;
        const std::vector<int32_t>& indices = m_desc.faceVertexIndices;
        std::vector<Float3> faces;
        std::map<std::pair<int32_t, uint32_t>, Float3> sums;
        size_t corner = 0;
        for (size_t face = 0; face < m_smoothing.size(); ++face) {
            const size_t count = size_t(m_desc.faceVertexCounts[face]);
            Float3 area;
            for (size_t i = 0; i < count; ++i) {
                const Float3 a = points[size_t(indices[corner + i])];
                const Float3 b = points[size_t(indices[corner + (i + 1) % count])];
                const Float3 edge = Cross(a, b);
                area = {area.x + edge.x, area.y + edge.y, area.z + edge.z};
            }
            faces.push_back(area);
            if (m_smoothing[face] != 0) {
                for (size_t i = 0; i < count; ++i) {
                    Float3& sum = sums[{indices[corner + i], m_smoothing[face]}];
                    sum = {sum.x + area.x, sum.y + area.y, sum.z + area.z};
                }
            }
            corner += count;
        }
        std::vector<Float3> normals;
        normals.reserve(indices.size());
        corner = 0;
        for (size_t face = 0; face < m_smoothing.size(); ++face) {
            const size_t count = size_t(m_desc.faceVertexCounts[face]);
            for (size_t i = 0; i < count; ++i) {
                normals.push_back(Normalize(
                    m_smoothing[face] == 0 ? faces[face]
                                           : sums.at({indices[corner + i], m_smoothing[face]})));
            }
            corner += count;
        }
        return normals;
    }

    const tinyobj::attrib_t& m_attributes;
    const std::filesystem::path& m_file;
    Mesh::Desc m_desc;
    std::unordered_map<int, int32_t> m_points;
    std::unordered_map<int, int32_t> m_stSlots;
    std::vector<Float2> m_stValues;
    std::vector<int32_t> m_stIndices;
    std::vector<Float3> m_colorValues;
    std::vector<uint32_t> m_smoothing;
    bool m_normals = true;
    bool m_st = true;
    bool m_colors = false;
};

std::string_view ScalarChannel(const tinyobj::texture_option_t& option) {
    switch (option.imfchan) {
    case 'g':
        return "g";
    case 'b':
        return "b";
    default:
        return "r";
    }
}

std::string Text(const std::vector<std::byte>& bytes) {
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

class SourceMaterials : public tinyobj::MaterialReader {
  public:
    explicit SourceMaterials(const ImportSource& source) : m_source(source) {}

    bool operator()(const std::string& name, std::vector<tinyobj::material_t>* materials,
                    std::map<std::string, int>* ids, std::string* warning,
                    std::string* error) override {
        FileBytes bytes;
        try {
            bytes = m_source.Read(name);
        } catch (const Error& missing) {
            *warning += std::string(missing.what()) + "\n";
            return false;
        }
        std::istringstream text(Text(*bytes));
        tinyobj::LoadMtl(ids, materials, &text, warning, error);
        return true;
    }

  private:
    const ImportSource& m_source;
};

struct LookTexture {
    std::string input;
    std::string output;
    size_t placement = 0;
    UsdUVTexture::Desc texture;
};

struct Look {
    UsdPreviewSurface::Desc surface;
    std::vector<LookTexture> textures;
    std::vector<TexturePlacement::Desc> placements;
};

void DefineLook(Scene& scene, std::string_view at, const Look& look) {
    const std::string root(at);
    const Prim material = scene.Define<Material>(root);
    const Prim surface = scene.Define<UsdPreviewSurface>(root + "/Surface", look.surface);
    material.Connect("surface", surface, "surface");
    std::vector<Prim> placements;
    for (size_t index = 0; index < look.placements.size(); ++index) {
        const std::string name = index == 0 ? "Placement" : std::format("Placement_{}", index + 1);
        placements.push_back(
            scene.Define<TexturePlacement>(root + "/" + name, look.placements[index]));
    }
    for (const LookTexture& texture : look.textures) {
        std::string name = texture.input;
        name[0] = char(std::toupper(static_cast<unsigned char>(name[0])));
        const Prim read =
            scene.Define<UsdUVTexture>(root + "/" + name + "Texture", texture.texture);
        read.Connect("st", placements[texture.placement], "st");
        surface.Connect(texture.input, read, texture.output);
    }
}

class SurfaceBuilder {
  public:
    SurfaceBuilder(const tinyobj::material_t& material, std::string_view path,
                   const ImportSource& source, ImportPlan& plan)
        : m_material(material), m_path(path), m_source(source), m_plan(plan) {}

    Look Build() && {
        const tinyobj::material_t& m = m_material;
        UsdPreviewSurface::Desc& surface = m_look.surface;
        surface.diffuseColor = {m.diffuse[0], m.diffuse[1], m.diffuse[2]};
        Map("diffuseColor", "map_Kd", m.diffuse_texname, m.diffuse_texopt);
        surface.emissiveColor = {m.emission[0], m.emission[1], m.emission[2]};
        Map("emissiveColor", "map_Ke", m.emissive_texname, m.emissive_texopt);
        surface.opacity = m.dissolve;
        Map("opacity", "map_d", m.alpha_texname, m.alpha_texopt,
            m.alpha_texname == m.diffuse_texname ? "a" : ScalarChannel(m.alpha_texopt));
        surface.roughness = m.roughness > 0   ? m.roughness
                            : m.shininess > 0 ? std::pow(2 / (m.shininess + 2), 0.25f)
                                              : 0.5f;
        Map("roughness", "map_Pr", m.roughness_texname, m.roughness_texopt,
            ScalarChannel(m.roughness_texopt));
        surface.metallic = m.metallic;
        Map("metallic", "map_Pm", m.metallic_texname, m.metallic_texopt,
            ScalarChannel(m.metallic_texopt));
        if (m.ior > 1) {
            surface.ior = m.ior;
        }
        if (m.clearcoat_thickness > 0) {
            surface.clearcoat = m.clearcoat_thickness;
            surface.clearcoatRoughness = m.clearcoat_roughness;
        }
        if (Map("normal", "norm", m.normal_texname, m.normal_texopt)) {
            surface.normalStrength = m.normal_texopt.bump_multiplier;
        } else if (Map("normal", "map_Bump", m.bump_texname, m.bump_texopt)) {
            Log(LogLevel::Debug, "{} reads its map_Bump \"{}\" as a normal map", m_path,
                m.bump_texname);
            surface.normalStrength = m.bump_texopt.bump_multiplier;
        }
        Ignore("map_Ka", m.ambient_texname);
        Ignore("map_Ks", m.specular_texname);
        Ignore("map_Ns", m.specular_highlight_texname);
        Ignore("disp", m.displacement_texname);
        Ignore("refl", m.reflection_texname);
        Ignore("map_Ps", m.sheen_texname);
        if (m.sheen > 0) {
            m_ignored.push_back("Ps");
        }
        if (m.anisotropy > 0) {
            m_ignored.push_back("aniso");
        }
        if (!m_ignored.empty()) {
            Log(LogLevel::Debug, "{} ignores {}, which UsdPreviewSurface has no input for", m_path,
                Join(m_ignored));
        }
        return std::move(m_look);
    }

  private:
    bool Map(std::string_view input, std::string_view statement, const std::string& name,
             const tinyobj::texture_option_t& option, std::string_view output = "rgb") {
        if (name.empty()) {
            return false;
        }
        std::string file;
        if (FileBytes bytes = m_source.Find(name)) {
            file = m_plan.AddFile(name, std::move(bytes));
        } else {
            const std::filesystem::path located = m_source.Locate(name);
            if (!std::filesystem::exists(located)) {
                Log(LogLevel::Warning, "{}'s {} \"{}\" doesn't exist; {} stays untextured", m_path,
                    statement, located.string(), input);
                return false;
            }
            file = located.generic_string();
        }
        const bool color = input == "diffuseColor" || input == "emissiveColor";
        LookTexture texture{
            .input = std::string(input), .output = std::string(output), .placement = Place(option)};
        texture.texture.file = std::move(file);
        texture.texture.sourceColorSpace = color ? TextureColorSpace::SRGB : TextureColorSpace::Raw;
        if (option.clamp) {
            texture.texture.wrapS = TextureWrapMode::Clamp;
            texture.texture.wrapT = TextureWrapMode::Clamp;
        }
        if (input == "normal") {
            texture.texture.scale = {2, 2, 2, 1};
            texture.texture.bias = {-1, -1, -1, 0};
        }
        m_look.textures.push_back(std::move(texture));
        return true;
    }

    size_t Place(const tinyobj::texture_option_t& option) {
        const TexturePlacement::Desc placement{
            .uvScale = {option.scale[0], option.scale[1]},
            .uvOffset = {option.origin_offset[0], option.origin_offset[1]}};
        std::vector<TexturePlacement::Desc>& placements = m_look.placements;
        const auto found = std::ranges::find_if(placements, [&](const TexturePlacement::Desc& p) {
            return p.uvScale == placement.uvScale && p.uvOffset == placement.uvOffset;
        });
        if (found != placements.end()) {
            return size_t(found - placements.begin());
        }
        placements.push_back(placement);
        return placements.size() - 1;
    }

    void Ignore(std::string_view statement, const std::string& name) {
        if (!name.empty()) {
            m_ignored.emplace_back(statement);
        }
    }

    const tinyobj::material_t& m_material;
    std::string m_path;
    const ImportSource& m_source;
    ImportPlan& m_plan;
    Look m_look;
    std::vector<std::string> m_ignored;
};

struct ObjData {
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
};

ObjData Parse(const ImportSource& source) {
    const std::filesystem::path& file = source.file;
    std::istringstream text(Text(*source.ReadEntry()));
    SourceMaterials reader(source);
    ObjData data;
    std::string warnings;
    std::string errors;
    if (!tinyobj::LoadObj(&data.attributes, &data.shapes, &data.materials, &warnings, &errors,
                          &text, &reader, false, false)) {
        throw Error("Reading \"{}\" as OBJ: {}", file.string(), errors);
    }
    const std::string messages = warnings + errors;
    for (const std::string_view warning : Split(messages, '\n')) {
        if (!warning.empty()) {
            Log(LogLevel::Warning, "reading \"{}\": {}", file.string(), warning);
        }
    }
    Log(LogLevel::Debug, "\"{}\" has {} shapes, {} materials and {} vertices", file.string(),
        data.shapes.size(), data.materials.size(), data.attributes.vertices.size() / 3);
    return data;
}

std::vector<std::string> DefineMaterials(ImportPlan& plan, std::string_view path,
                                         const ObjData& data, const ImportSource& source,
                                         ImportContext& context, ChildNames& children) {
    std::vector<std::string> looks;
    if (data.materials.empty()) {
        return looks;
    }
    const std::string scope = std::format("{}/{}", path, children.Take("Looks", "Looks"));
    plan.Define<Scope>(scope);
    ChildNames names;
    for (const tinyobj::material_t& material : data.materials) {
        context.Report("materials", looks.size(), data.materials.size());
        const std::string look = std::format("{}/{}", scope, names.Take(material.name, "Material"));
        plan.Add(look, [built = SurfaceBuilder(material, look, source, plan).Build()](
                           Scene& scene, std::string_view at) { DefineLook(scene, at, built); });
        Log(LogLevel::Debug, "read OBJ material \"{}\" for {}", material.name, look);
        looks.push_back(look);
    }
    context.Report("materials", looks.size(), data.materials.size());
    return looks;
}

void DefineMeshes(ImportPlan& plan, std::string_view path, const ObjData& data,
                  std::span<const std::string> looks, const ImportSource& source,
                  ImportContext& context, ChildNames& children) {
    const std::vector<tinyobj::shape_t>& shapes = data.shapes;
    for (size_t index = 0; index < shapes.size(); ++index) {
        context.Report("shapes", index, shapes.size());
        const tinyobj::shape_t& shape = shapes[index];
        if (!shape.lines.indices.empty() || !shape.points.indices.empty()) {
            Log(LogLevel::Warning,
                "\"{}\" skips the lines and points of shape \"{}\", which import as nothing yet",
                source.file.string(), shape.name);
        }
        if (shape.mesh.num_face_vertices.empty()) {
            continue;
        }
        MeshBuilder builder(data.attributes, source.file);
        std::map<int, std::vector<int32_t>> faces;
        size_t corner = 0;
        for (size_t face = 0; face < shape.mesh.num_face_vertices.size(); ++face) {
            const size_t count = shape.mesh.num_face_vertices[face];
            const int material =
                face < shape.mesh.material_ids.size() ? shape.mesh.material_ids[face] : -1;
            const bool bound = material >= 0 && size_t(material) < looks.size();
            faces[bound ? material : -1].push_back(int32_t(face));
            const uint32_t smoothing = face < shape.mesh.smoothing_group_ids.size()
                                           ? shape.mesh.smoothing_group_ids[face]
                                           : 0;
            builder.AddFace(std::span(shape.mesh.indices).subspan(corner, count), smoothing);
            corner += count;
        }
        const bool single = faces.size() <= 1;
        const int only = faces.empty() ? -1 : faces.begin()->first;
        const std::string mesh = fmt::format("{}/{}", path, children.Take(shape.name, "Mesh"));
        const uint32_t faceCount = builder.FaceCount();
        const uint32_t pointCount = builder.PointCount();
        plan.Define<Mesh>(
            mesh, std::move(builder).Finish(single && only >= 0 ? looks[size_t(only)] : ""));
        Log(LogLevel::Debug, "read OBJ shape \"{}\" for {} with {} faces and {} points", shape.name,
            mesh, faceCount, pointCount);
        if (single) {
            continue;
        }
        ChildNames subsets;
        for (auto& [material, indices] : faces) {
            if (material < 0) {
                continue;
            }
            const std::string subset = fmt::format(
                "{}/{}", mesh, subsets.Take(data.materials[size_t(material)].name, "Subset"));
            plan.Define<GeomSubset>(subset, {.materialBinding = {looks[size_t(material)]},
                                             .indices = std::move(indices),
                                             .familyName = "materialBind"});
        }
    }
    context.Report("shapes", shapes.size(), shapes.size());
}

class ObjImporter final : public Importer {
  public:
    std::string Name() const override {
        return "OBJ";
    }

    std::vector<std::string> Extensions() const override {
        return {".obj"};
    }

    ImportPlan Read(std::string_view path, const ImportSource& source,
                    ImportContext& context) override {
        ERNIE_ZONE("Import/Obj");
        context.Report("parsing", 0, 1);
        const ObjData data = Parse(source);
        context.Report("parsing", 1, 1);

        const double half = std::sqrt(0.5);
        const float units = context.MetersPerUnit().value_or(1);
        const bool yUp = context.Up().value_or(UpAxis::Y) == UpAxis::Y;
        ImportPlan plan;
        plan.Define<Xform>(std::string(path), {.orient = yUp ? Quatd{half, 0, 0, half} : Quatd{},
                                               .scale = {units, units, units}});
        ChildNames children;
        const std::vector<std::string> looks =
            DefineMaterials(plan, path, data, source, context, children);
        DefineMeshes(plan, path, data, looks, source, context, children);
        return plan;
    }
};

}

std::shared_ptr<Importer> CreateObjImporter() {
    return std::make_shared<ObjImporter>();
}

}

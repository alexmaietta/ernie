// SPDX-License-Identifier: BSD-3-Clause
#include "bench.h"

#include "ernie/scene/material_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ernie::bench {

namespace {

template <typename Info>
const Info& Find(const std::vector<Info>& registry,
                 std::string_view name, std::string_view kind) {
    for (const Info& info : registry) {
        if (info.name == name) {
            return info;
        }
    }
    std::string known;
    for (const Info& info : registry) {
        known += (known.empty() ? "" : ", ") + info.name;
    }
    throw std::invalid_argument("No " + std::string(kind) +
                                " named " + std::string(name) +
                                "; there are " + known);
}

}  // namespace

std::string Case::Name() const {
    std::string name = scenario;
    // The scenario's order, then any others.
    std::vector<std::string> order;
    for (const Param& param : ParamsOf(scenario)) {
        order.push_back(param.name);
    }
    for (const auto& [key, value] : params) {
        if (std::find(order.begin(), order.end(), key) ==
            order.end()) {
            order.push_back(key);
        }
    }
    for (const std::string& key : order) {
        if (const auto value = params.find(key);
            value != params.end()) {
            name += "/" + key + "=" + value->second;
        }
    }
    return name;
}

std::string_view NameOf(Scope scope) {
    switch (scope) {
    case Scope::Operation:
        return "operation";
    case Scope::Subsystem:
        return "subsystem";
    case Scope::Workflow:
        return "workflow";
    }
    return "";
}

std::string_view NameOf(Completion completion) {
    return completion == Completion::Fenced ? "fenced" : "pipelined";
}

Scope ScopeNamed(std::string_view name) {
    for (const Scope scope :
         {Scope::Operation, Scope::Subsystem, Scope::Workflow}) {
        if (NameOf(scope) == name) {
            return scope;
        }
    }
    throw std::invalid_argument(
        "No scope named " + std::string(name) +
        "; there are operation, subsystem, workflow");
}

Completion CompletionNamed(std::string_view name) {
    for (const Completion completion :
         {Completion::Fenced, Completion::Pipelined}) {
        if (NameOf(completion) == name) {
            return completion;
        }
    }
    throw std::invalid_argument("No completion named " +
                                std::string(name) +
                                "; there are fenced, pipelined");
}

Definition Scenario::Describe(const Case&) const {
    return {};
}

std::string Scenario::Validate(const Case&,
                               const RenderProductSetOutputs&) {
    return {};
}

void Scenario::Teardown(Renderer&, Scene&, const Case&) {}

void Scenario::Setup(Renderer&, Scene&, const Case&) {}

void Scenario::Step(Renderer&, Scene&, const Case&, int) {}

std::vector<Path> Scenario::Products(const Case&) const {
    return {kPaths.product};
}

std::vector<Unit> Scenario::Units(const Case&) const {
    return {};
}

std::string Scenario::Get(const Case& c,
                          std::string_view name) const {
    for (const Param& param : Params()) {
        if (param.name == name) {
            const auto set = c.params.find(param.name);
            return set == c.params.end() ? param.defaultValue
                                         : set->second;
        }
    }
    throw std::invalid_argument("Scenario " + c.scenario +
                                " has no parameter " +
                                std::string(name));
}

double Scenario::GetDouble(const Case& c,
                           std::string_view name) const {
    return std::stod(Get(c, name));
}

int Scenario::GetInt(const Case& c, std::string_view name) const {
    return std::stoi(Get(c, name));
}

void Scenario::AddStat(const std::string& name, double value) {
    for (auto& [kept, total] : stats_) {
        if (kept == name) {
            total += value;
            return;
        }
    }
    stats_.emplace_back(name, value);
}

void Scenario::DefineInstances(
    Scene& scene, const Path& path,
    const std::vector<Path>& prototypes,
    const std::vector<Placement>& placements) {
    const auto n = static_cast<std::int64_t>(placements.size());
    std::vector<float> positions;
    std::vector<float> orientations;
    std::vector<float> scales;
    std::vector<float> colours;
    std::vector<std::int32_t> protoIndices;
    std::vector<std::int32_t> worlds;
    for (const Placement& p : placements) {
        positions.insert(positions.end(),
                         {p.position.x, p.position.y, p.position.z});
        // x, y, z, w: USD's memory layout.
        orientations.insert(orientations.end(),
                            {p.orientation.x, p.orientation.y,
                             p.orientation.z, p.orientation.w});
        scales.insert(scales.end(),
                      {p.scale.x, p.scale.y, p.scale.z});
        colours.insert(colours.end(),
                       {p.colour.r, p.colour.g, p.colour.b});
        protoIndices.push_back(p.proto);
        worlds.push_back(p.world);
    }
    PointInstancer instancer =
        PointInstancer::Define(scene, path)
            .SetPositions(Tensor::Copy<float>(positions, {n, 3}))
            .SetOrientations(
                Tensor::Copy<float>(orientations, {n, 4}))
            .SetScales(Tensor::Copy<float>(scales, {n, 3}))
            .SetProtoIndices(
                Tensor::Copy<std::int32_t>(protoIndices, {n}))
            .SetWorlds(Tensor::Copy<std::int32_t>(worlds, {n}));
    if (!placements.empty() && placements.front().colour.r >= 0.0f) {
        instancer.SetDisplayColor(
            Tensor::Copy<float>(colours, {n, 3}));
    }
    scene.Set(path, tokens::kPrototypes, prototypes);

    // World 0's triangles stand for each world's.
    std::vector<double> triangles;
    for (const Path& prototype : prototypes) {
        const double count = static_cast<double>(
            MeshDataOf(scene, prototype).indices.size() / 3);
        triangles.push_back(count);
        AddStat("unique_triangles", count);
    }
    double world0 = 0.0;
    for (const Placement& p : placements) {
        if (p.world <= 0) {
            world0 += triangles[static_cast<std::size_t>(p.proto)];
        }
    }
    AddStat("instances", static_cast<double>(n));
    AddStat("triangles_per_world", world0);
}

glm::quat Yaw(float radians) {
    return glm::angleAxis(radians, glm::vec3(0.0f, 0.0f, 1.0f));
}

std::vector<Case> Sweep::Cases() const {
    std::vector<Case> cases;
    for (const std::string& scenario : scenarios) {
        const std::vector<Param> params = ParamsOf(scenario);
        const auto has = [&](const std::string& name) {
            return std::any_of(params.begin(), params.end(),
                               [&](const Param& param) {
                                   return param.name == name;
                               });
        };
        std::vector<Case> partial{{.scenario = scenario}};
        for (const auto& [name, values] : axes) {
            if (!has(name)) {
                continue;
            }
            std::vector<Case> next;
            for (const Case& c : partial) {
                for (const std::string& value : values) {
                    Case more = c;
                    more.params[name] = value;
                    next.push_back(std::move(more));
                }
            }
            partial = std::move(next);
        }
        Append(cases, partial);
    }
    return cases;
}

std::vector<std::string> Values(std::initializer_list<int> values) {
    std::vector<std::string> strings;
    for (const int value : values) {
        strings.push_back(std::to_string(value));
    }
    return strings;
}

void Append(std::vector<Case>& cases, const std::vector<Case>& more) {
    for (const Case& c : more) {
        const bool repeat = std::any_of(
            cases.begin(), cases.end(), [&](const Case& other) {
                return other.scenario == c.scenario &&
                       other.params == c.params;
            });
        if (!repeat) {
            cases.push_back(c);
        }
    }
}

std::vector<ScenarioInfo>& Scenarios() {
    static std::vector<ScenarioInfo> registry;
    return registry;
}

std::vector<PresetInfo>& Presets() {
    static std::vector<PresetInfo> registry;
    return registry;
}

std::vector<OutputsInfo>& OutputSets() {
    static std::vector<OutputsInfo> registry;
    return registry;
}

std::vector<SuiteInfo>& Suites() {
    static std::vector<SuiteInfo> registry;
    return registry;
}

const ScenarioInfo& FindScenario(std::string_view name) {
    return Find(Scenarios(), name, "scenario");
}

const PresetInfo& FindPreset(std::string_view name) {
    return Find(Presets(), name, "preset");
}

const OutputsInfo& FindOutputs(std::string_view name) {
    return Find(OutputSets(), name, "output set");
}

const SuiteInfo& FindSuite(std::string_view name) {
    return Find(Suites(), name, "suite");
}

std::vector<Param> ParamsOf(std::string_view name) {
    return FindScenario(name).make()->Params();
}

Registration::Registration(ScenarioInfo info) {
    Scenarios().push_back(std::move(info));
}

Registration::Registration(PresetInfo info) {
    Presets().push_back(std::move(info));
}

Registration::Registration(OutputsInfo info) {
    OutputSets().push_back(std::move(info));
}

Registration::Registration(SuiteInfo info) {
    Suites().push_back(std::move(info));
}

Mesh DefineMesh(Scene& scene, const Path& path,
                const MeshData& mesh) {
    const auto points =
        static_cast<std::int64_t>(mesh.points.size() / 3);
    const auto corners =
        static_cast<std::int64_t>(mesh.indices.size());
    const std::vector<std::int32_t> counts(
        static_cast<std::size_t>(corners / 3), 3);
    const std::vector<std::int32_t> indices(mesh.indices.begin(),
                                            mesh.indices.end());
    Mesh prim =
        Mesh::Define(scene, path)
            .SetPoints(Tensor::Copy<float>(mesh.points, {points, 3}))
            .SetFaceVertexCounts(
                Tensor::Copy<std::int32_t>(counts, {corners / 3}))
            .SetFaceVertexIndices(
                Tensor::Copy<std::int32_t>(indices, {corners}));
    if (!mesh.normals.empty()) {
        prim.SetNormals(
            Tensor::Copy<float>(mesh.normals, {points, 3}));
    }
    if (!mesh.uvs.empty()) {
        // One per corner: faceVarying, by count.
        scene.Write(path, "primvars:st",
                    Tensor::Copy<float>(mesh.uvs, {corners, 2}));
    }
    return prim;
}

void ProjectUvs(MeshData& mesh, float perMetre) {
    mesh.uvs.clear();
    const auto point = [&](std::uint32_t i) {
        return glm::vec3(mesh.points[3 * i], mesh.points[3 * i + 1],
                         mesh.points[3 * i + 2]);
    };
    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const glm::vec3 a = point(mesh.indices[i]);
        const glm::vec3 b = point(mesh.indices[i + 1]);
        const glm::vec3 c = point(mesh.indices[i + 2]);
        const glm::vec3 face = glm::abs(glm::cross(b - a, c - a));
        const int axis = face.x > face.y && face.x > face.z ? 0
                         : face.y > face.z                  ? 1
                                                            : 2;
        const int u = axis == 0 ? 1 : 0;
        const int v = axis == 2 ? 1 : 2;
        for (const glm::vec3& p : {a, b, c}) {
            mesh.uvs.insert(mesh.uvs.end(),
                            {p[u] * perMetre, p[v] * perMetre});
        }
    }
}

void DefineTexturedMaterial(Scene& scene, const Path& path,
                            const std::vector<std::uint8_t>& texels,
                            int size) {
    const Path surface = path.Child("Surface");
    const Path texture = path.Child("Tex");
    const Path st = path.Child("St");
    Material::Define(scene, path);
    Shader::Define(scene, surface)
        .SetInfoId(tokens::kUsdPreviewSurface)
        .SetRoughness(0.7);
    Shader::Define(scene, texture)
        .SetInfoId(tokens::kUsdUVTexture)
        .SetTexture(
            Tensor::Copy<std::uint8_t>(texels, {size, size, 4}));
    Shader::Define(scene, st)
        .SetInfoId(tokens::kUsdPrimvarReaderFloat2)
        .SetVarname("st");
    Connect(scene, path, tokens::kSurfaceOutput, surface,
            tokens::kSurfaceOutput);
    Connect(scene, texture, tokens::kSt, st, "outputs:result");
    Connect(scene, surface, tokens::kDiffuseColor, texture,
            "outputs:rgb");
}

std::vector<std::uint8_t> CheckerTexture(int size, int cells,
                                         glm::vec3 a, glm::vec3 b,
                                         float noise,
                                         std::uint32_t seed) {
    // Value noise on an 8 x 8 lattice, bilinearly interpolated.
    constexpr int kLattice = 8;
    Random random(seed);
    std::vector<float> lattice((kLattice + 1) * (kLattice + 1));
    for (float& value : lattice) {
        value = random.Uniform(-1.0f, 1.0f);
    }
    const auto at = [&](int i, int j) {
        return lattice[static_cast<std::size_t>(j * (kLattice + 1) +
                                                i)];
    };
    std::vector<std::uint8_t> texels(static_cast<std::size_t>(size) *
                                     static_cast<std::size_t>(size) *
                                     4);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const bool odd =
                (x * cells / size + y * cells / size) % 2 != 0;
            const float u = static_cast<float>(x * kLattice) /
                            static_cast<float>(size);
            const float v = static_cast<float>(y * kLattice) /
                            static_cast<float>(size);
            const int i = static_cast<int>(u);
            const int j = static_cast<int>(v);
            const float fu = u - static_cast<float>(i);
            const float fv = v - static_cast<float>(j);
            const float n = glm::mix(
                glm::mix(at(i, j), at(i + 1, j), fu),
                glm::mix(at(i, j + 1), at(i + 1, j + 1), fu), fv);
            const glm::vec3 colour = glm::clamp(
                (odd ? b : a) * (1.0f + noise * n), 0.0f, 1.0f);
            std::uint8_t* texel =
                &texels[(static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(size) +
                         static_cast<std::size_t>(x)) *
                        4];
            for (int c = 0; c < 3; ++c) {
                texel[c] =
                    static_cast<std::uint8_t>(colour[c] * 255.0f);
            }
            texel[3] = 255;
        }
    }
    return texels;
}

glm::dmat4 LookAt(glm::dvec3 eye, glm::dvec3 target) {
    return LookAtTransform(eye, target, {0.0, 0.0, 1.0});
}

std::uint32_t Random::Next() {
    // PCG-XSH-RR: M. E. O'Neill, "PCG: A Family of Simple Fast
    // Space-Efficient Statistically Good Algorithms for Random Number
    // Generation", 2014.
    const std::uint64_t old = state_;
    state_ = old * 6364136223846793005ull + 1442695040888963407ull;
    const auto shifted =
        static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
    const auto rotation = static_cast<std::uint32_t>(old >> 59u);
    return (shifted >> rotation) |
           (shifted << ((32u - rotation) & 31u));
}

float Random::Uniform(float lo, float hi) {
    return lo + (hi - lo) * static_cast<float>(Next() >> 8) *
                    (1.0f / 16777216.0f);
}

int Random::Int(int lo, int hiInclusive) {
    return lo + static_cast<int>(Next() % static_cast<std::uint32_t>(
                                              hiInclusive - lo + 1));
}

}  // namespace ernie::bench

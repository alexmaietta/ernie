// SPDX-License-Identifier: BSD-3-Clause
/// @file validate.cpp
/// @brief Checking scene files: loading's problems, with near-miss suggestions, and relationships
/// to missing prims.
#include "ernie/scene/reference.h"
#include "ernie/scene/scene.h"
#include "ernie/util/text.h"

#include <limits>

namespace ernie {
namespace {

/// @return Every attribute name the prim's type and API schemas declare.
std::vector<std::string> DeclaredNames(const Scene& scene, PrimId prim) {
    std::vector<std::string> names;
    const auto add = [&](const ApiName& api) {
        for (const AttributeDef* def : api.schema->AllAttributes()) {
            names.push_back(api.Attribute(def->name));
        }
    };
    if (const SchemaDef* schema = scene.Schema(prim)) {
        add({schema, {}});
        for (const std::string_view builtin : schema->AllBuiltinApis()) {
            add(ApiName::Parse(builtin));
        }
    }
    for (const std::string& api : scene.AppliedApis(prim)) {
        add(ApiName::Parse(api));
    }
    return names;
}

/// @return The declared name nearest a misspelt one, if near enough to be meant.
std::string Nearest(const std::vector<std::string>& names, std::string_view name) {
    std::string nearest;
    std::size_t best = std::numeric_limits<std::size_t>::max();
    for (const std::string& candidate : names) {
        if (const std::size_t distance = EditDistance(candidate, name); distance < best) {
            best = distance;
            nearest = candidate;
        }
    }
    return best <= std::max<std::size_t>(2, name.size() / 3) ? nearest : std::string();
}

void Suggest(const Scene& scene, SceneProblem& problem) {
    const std::optional<PrimId> prim =
        problem.path.empty() ? std::nullopt : scene.Find(Path(problem.path));
    if (!prim || problem.what.find("has no attribute") == std::string::npos) {
        return;
    }
    if (const std::string nearest = Nearest(DeclaredNames(scene, *prim), problem.key);
        !nearest.empty()) {
        problem.what += "; did you mean '" + nearest + "'?";
    }
}

std::vector<Path> Targets(const Value& value) {
    if (const auto* path = value.Get<Path>()) {
        return path->IsEmpty() ? std::vector<Path>{} : std::vector<Path>{*path};
    }
    if (const auto* paths = value.Get<std::vector<Path>>()) {
        return *paths;
    }
    return {};
}

/// Adds a problem for each relationship of the prim targeting a prim the scene doesn't have.
void CheckTargets(const Scene& scene, const Path& path, std::vector<SceneProblem>& problems) {
    const PrimId prim = *scene.Find(path);
    for (const std::string& name : scene.Attributes(path)) {
        const std::optional<Value> value = scene.Get(prim, name);
        for (const Path& target : value ? Targets(*value) : std::vector<Path>{}) {
            // A property target, as a connection's, names its prim before the dot.
            const std::string text = target.String();
            if (!scene.Has(Path(text.substr(0, text.find('.'))))) {
                problems.push_back({path.String(), name, "targets " + text + ", which is no prim"});
            }
        }
    }
}

}  // namespace

std::vector<SceneProblem> ValidateScene(std::string_view json) {
    Scene scene;
    std::vector<SceneProblem> problems;
    LoadScene(scene, json, {}, &problems);
    for (SceneProblem& problem : problems) {
        Suggest(scene, problem);
    }
    for (const Path& path : scene.Prims()) {
        CheckTargets(scene, path, problems);
    }
    return problems;
}

}  // namespace ernie

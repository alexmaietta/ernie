// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ernie/app/viewer/shell/history.h"
#include "ernie/app/viewer/shell/selection.h"
#include "ernie/scene/scene.h"

namespace ernie::app {

enum class PrimGroup { Organize, Geometry, Lights, Cameras, Shading, Render, Other };

inline constexpr const char* kPrimGroupNames[] = {"Organize",  "Geometry",  "Lights", "Cameras",
                                                  "Materials", "Rendering", "Other"};

PrimGroup Classify(SchemaId schema);

class Editor {
  public:
    using Target = std::function<View()>;

    Editor(Ref<Scene> scene, Selection& selection, History& history)
        : m_scene(std::move(scene)), m_selection(selection), m_history(history) {}

    std::string Create(SchemaId type, const Double3& at, const Quatd& orient);
    void SetVisible(const std::string& path, bool visible);
    void ApplyApi(SchemaId api, const std::string& instance = {});
    void SetCells(std::string name, Target target, ColumnRef column, std::vector<std::byte> before,
                  std::vector<std::byte> after);
    void SetTargets(std::string name, Target target, ColumnRef column,
                    std::vector<std::vector<std::string>> before, std::vector<std::string> after);
    void SetPose(std::string name, const std::string& path, Double3 translateBefore,
                 Quatd orientBefore, Float3 scaleBefore, Double3 translateAfter, Quatd orientAfter,
                 Float3 scaleAfter);
    bool Rename(const std::string& path, std::string_view name);
    bool Reparent(const std::string& path, const std::string& parent);
    void Duplicate();
    void Delete();

  private:
    std::vector<std::string> SelectedRoots() const;
    std::string FreeSibling(const std::string& path) const;
    void Retarget(const std::string& from, const std::string& to);

    Ref<Scene> m_scene;
    Selection& m_selection;
    History& m_history;
};

}

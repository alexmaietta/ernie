// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/shell/editor.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <utility>

#include <fmt/format.h>

#include "ernie/util/log.h"

namespace ernie::app {
namespace {

std::string Leaf(const std::string& path) {
    return path.substr(path.rfind('/') + 1);
}

std::string Parent(const std::string& path) {
    return path.substr(0, path.rfind('/'));
}

bool Below(const std::string& path, const std::string& ancestor) {
    return path.size() > ancestor.size() && path.starts_with(ancestor) &&
           path[ancestor.size()] == '/';
}

}

PrimGroup Classify(SchemaId schema) {
    const std::string_view name = Info(schema).name;
    if (name == "Xform" || name == "Scope" || name == "World") {
        return PrimGroup::Organize;
    }
    for (SchemaId level = schema; level != SchemaId::None; level = Info(level).base) {
        const std::string_view base = Info(level).name;
        if (base == "Gprim") {
            return PrimGroup::Geometry;
        }
        if (base.ends_with("LightBase")) {
            return PrimGroup::Lights;
        }
        if (base == "Camera") {
            return PrimGroup::Cameras;
        }
        if (base == "Connectable") {
            return PrimGroup::Shading;
        }
        if (base == "Background" || base.starts_with("Render")) {
            return PrimGroup::Render;
        }
    }
    return PrimGroup::Other;
}

std::vector<std::string> Editor::SelectedRoots() const {
    std::vector<std::string> roots;
    for (const std::string& path : m_selection.paths) {
        const bool nested = std::ranges::any_of(
            m_selection.paths, [&](const std::string& other) { return Below(path, other); });
        if (!nested && m_scene->Find(path)) {
            roots.push_back(path);
        }
    }
    return roots;
}

std::string Editor::FreeSibling(const std::string& path) const {
    for (int copy = 1;; ++copy) {
        std::string candidate = fmt::format("{}_{}", path, copy);
        if (!m_scene->Find(candidate)) {
            return candidate;
        }
    }
}

void Editor::Retarget(const std::string& from, const std::string& to) {
    for (std::string& path : m_selection.paths) {
        if (path == from) {
            path = to;
        } else if (Below(path, from)) {
            path = to + path.substr(from.size());
        }
    }
}

std::string Editor::Create(SchemaId type, const Double3& at, const Quatd& orient) {
    std::string parent;
    if (m_selection.paths.size() == 1) {
        if (const std::optional<Prim> selected = m_scene->Find(m_selection.paths.front());
            selected && Classify(selected->Schema()) == PrimGroup::Organize) {
            parent = m_selection.paths.front();
        }
    }
    const PrimGroup group = Classify(type);
    if (parent.empty() && group == PrimGroup::Shading && m_scene->Find("/Looks")) {
        parent = "/Looks";
    } else if (parent.empty() && group == PrimGroup::Render && m_scene->Find("/Render")) {
        parent = "/Render";
    }
    std::string path = parent + "/" + std::string(Info(type).name);
    if (m_scene->Find(path)) {
        path = FreeSibling(path);
    }
    const std::vector<std::string> before = m_selection.paths;
    const auto define = [this, type, path, at, orient] {
        const Prim prim = type == SchemaId::Material ? m_scene->DefineMaterial(path)
                                                     : m_scene->Define(path, type);
        if (prim.IsA(SchemaId::Xformable)) {
            prim.Set(Xformable::kTranslate, at);
            prim.Set(Xformable::kOrient, orient);
        }
        m_selection.Set(path);
    };
    m_history.Do({"Create " + Leaf(path),
                  [this, path, before] {
                      if (const std::optional<Prim> prim = m_scene->Find(path)) {
                          m_scene->Remove(*prim);
                      }
                      m_selection.paths = before;
                  },
                  define});
    return path;
}

void Editor::SetVisible(const std::string& path, bool visible) {
    const auto set = [scene = m_scene, path](bool shown) {
        scene->Select(path)
            .Get(Imageable::kVisibility)
            .Set(shown ? Visibility::Inherited : Visibility::Invisible);
    };
    m_history.Do({fmt::format("{} {}", visible ? "Show" : "Hide", Leaf(path)),
                  [set, visible] { set(!visible); }, [set, visible] { set(visible); }});
}

void Editor::SetCells(std::string name, Target target, ColumnRef column,
                      std::vector<std::byte> before, std::vector<std::byte> after) {
    const auto write = [target, column, name](const std::vector<std::byte>& cells, bool fill) {
        try {
            UntypedAttribute attribute = target().Get(column);
            if (fill) {
                attribute.Fill(cells);
            } else {
                attribute.Write(cells);
            }
        } catch (const std::exception& error) {
            Log(LogLevel::Warning, "{}: {}", name, error.what());
        }
    };
    m_history.Do(
        {name, [write, before] { write(before, false); }, [write, after] { write(after, true); }});
}

void Editor::SetPose(std::string name, const std::string& path, Double3 translateBefore,
                     Quatd orientBefore, Float3 scaleBefore, Double3 translateAfter,
                     Quatd orientAfter, Float3 scaleAfter) {
    const auto write = [scene = m_scene, path, name](Double3 translate, Quatd orient,
                                                     Float3 scale) {
        try {
            const View prim = scene->Select(path);
            prim.Get(Xformable::kTranslate).Set(translate);
            prim.Get(Xformable::kOrient).Set(orient);
            prim.Get(Xformable::kScale).Set(scale);
        } catch (const std::exception& error) {
            Log(LogLevel::Warning, "{}: {}", name, error.what());
        }
    };
    m_history.Push({std::move(name),
                    [write, translateBefore, orientBefore, scaleBefore] {
                        write(translateBefore, orientBefore, scaleBefore);
                    },
                    [write, translateAfter, orientAfter, scaleAfter] {
                        write(translateAfter, orientAfter, scaleAfter);
                    }});
}

void Editor::SetTargets(std::string name, Target target, ColumnRef column,
                        std::vector<std::vector<std::string>> before,
                        std::vector<std::string> after) {
    const auto undo = [target, column, name, before] {
        try {
            UntypedAttribute attribute = target().Get(column);
            for (uint32_t row = 0; row < before.size(); ++row) {
                attribute.SetTargets(row, before[row]);
            }
        } catch (const std::exception& error) {
            Log(LogLevel::Warning, "{}: {}", name, error.what());
        }
    };
    const auto redo = [target, column, name, after, rows = uint32_t(before.size())] {
        try {
            UntypedAttribute attribute = target().Get(column);
            for (uint32_t row = 0; row < rows; ++row) {
                attribute.SetTargets(row, after);
            }
        } catch (const std::exception& error) {
            Log(LogLevel::Warning, "{}: {}", name, error.what());
        }
    };
    m_history.Do({std::move(name), undo, redo});
}

bool Editor::Rename(const std::string& path, std::string_view name) {
    const std::string to = Parent(path) + "/" + std::string(name);
    if (to == path) {
        return true;
    }
    try {
        m_scene->Rename(*m_scene->Find(path), name);
    } catch (const std::exception& error) {
        Log(LogLevel::Warning, "Renaming {} to {}: {}", path, name, error.what());
        return false;
    }
    Retarget(path, to);
    const std::string old = Leaf(path);
    m_history.Push({fmt::format("Rename {} to {}", old, name),
                    [this, to, old, path] {
                        m_scene->Rename(*m_scene->Find(to), old);
                        Retarget(to, path);
                    },
                    [this, to, path, name = std::string(name)] {
                        m_scene->Rename(*m_scene->Find(path), name);
                        Retarget(path, to);
                    }});
    return true;
}

bool Editor::Reparent(const std::string& path, const std::string& parent) {
    const std::string to = parent + "/" + Leaf(path);
    if (to == path || parent == path || Below(parent, path)) {
        return false;
    }
    try {
        m_scene->Move(*m_scene->Find(path), to);
    } catch (const std::exception& error) {
        Log(LogLevel::Warning, "Moving {} under {}: {}", path, parent.empty() ? "/" : parent,
            error.what());
        return false;
    }
    Retarget(path, to);
    m_history.Push({fmt::format("Move {}", Leaf(path)),
                    [this, to, path] {
                        m_scene->Move(*m_scene->Find(to), path);
                        Retarget(to, path);
                    },
                    [this, to, path] {
                        m_scene->Move(*m_scene->Find(path), to);
                        Retarget(path, to);
                    }});
    return true;
}

void Editor::Duplicate() {
    struct Pasted {
        PrimCopy copy;
        std::string path;
    };

    auto pasted = std::make_shared<std::vector<Pasted>>();
    for (const std::string& path : SelectedRoots()) {
        pasted->push_back({m_scene->Copy(*m_scene->Find(path)), FreeSibling(path)});
    }
    if (pasted->empty()) {
        return;
    }
    const auto paste = [this, pasted] {
        m_selection.Clear();
        for (const Pasted& item : *pasted) {
            m_scene->Paste(item.copy, item.path);
            m_selection.paths.push_back(item.path);
        }
    };
    const std::vector<std::string> before = m_selection.paths;
    m_history.Do({pasted->size() == 1 ? "Duplicate " + Leaf(pasted->front().path)
                                      : fmt::format("Duplicate {} prims", pasted->size()),
                  [this, pasted, before] {
                      for (const Pasted& item : *pasted) {
                          if (const std::optional<Prim> prim = m_scene->Find(item.path)) {
                              m_scene->Remove(*prim);
                          }
                      }
                      m_selection.paths = before;
                  },
                  paste});
}

void Editor::Delete() {
    struct Removed {
        PrimCopy copy;
        std::string path;
    };

    auto removed = std::make_shared<std::vector<Removed>>();
    for (const std::string& path : SelectedRoots()) {
        removed->push_back({m_scene->Copy(*m_scene->Find(path)), path});
    }
    if (removed->empty()) {
        return;
    }
    const std::vector<std::string> before = m_selection.paths;
    const auto remove = [this, removed] {
        for (const Removed& item : *removed) {
            if (const std::optional<Prim> prim = m_scene->Find(item.path)) {
                m_scene->Remove(*prim);
            }
        }
        m_selection.Clear();
    };
    m_history.Do({removed->size() == 1 ? "Delete " + Leaf(removed->front().path)
                                       : fmt::format("Delete {} prims", removed->size()),
                  [this, removed, before] {
                      for (const Removed& item : *removed) {
                          m_scene->Paste(item.copy, item.path);
                      }
                      m_selection.paths = before;
                  },
                  remove});
}

}

// SPDX-License-Identifier: BSD-3-Clause
/// @file edit_history.h
/// @brief Undo and redo of scene edits, for the viewer.
///
/// References:
/// - E. Gamma, R. Helm, R. Johnson and J. Vlissides, "Design Patterns", the Command pattern,
///   Addison-Wesley, 1994.
#pragma once

#include "ernie/scene/scene.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie {

/// @brief Undo and redo for the edits made through it: attributes set or cleared, prims defined,
/// removed with their descendants, or copied in. Each step records what it changes, so `Undo`
/// puts it back and `Redo` does it again.
///
/// Continuous edits, as dragging a slider or a gizmo, merge into one step: an `open` edit of the
/// same attributes as the last step updates it until `Close`. Edits between `BeginGroup` and
/// `EndGroup` undo and redo together. Edits made to the scene directly are not recorded.
class EditHistory {
  public:
    /// @brief Sets attributes of a prim as one step, dropping the undone steps. With `open`, merges
    /// into the last step if that is an open edit of the same attributes.
    /// @throws std::invalid_argument as `Scene::Set`.
    void Set(Scene& scene, const Path& path, std::vector<std::pair<std::string, Value>> edits,
             bool open = false);
    void Set(Scene& scene, const Path& path, std::string_view attribute, Value value,
             bool open = false) {
        Set(scene, path, {{std::string(attribute), std::move(value)}}, open);
    }
    /// @brief Unauthors an attribute, back to its fallback.
    void Clear(Scene& scene, const Path& path, std::string_view attribute);
    /// @brief Defines a new prim; undoing removes it.
    /// @throws std::invalid_argument as `Scene::Define`, or if a prim is there already.
    void Define(Scene& scene, const Path& path, std::string_view type);
    /// @brief Removes a prim and its descendants; undoing restores them, sharing their arrays.
    /// @return Whether there was a prim.
    bool Remove(Scene& scene, const Path& path);
    /// @brief Copies a prim of `source` and its descendants to the same path of `scene`, as for
    /// prims authored in a scratch scene first; undoing removes them.
    /// @throws std::invalid_argument if `source` has no prim at `path` or `scene` has one.
    void Add(Scene& scene, const Scene& source, const Path& path);

    /// Groups the edits until `EndGroup` into one step. Groups nest; the outermost counts.
    void BeginGroup();
    void EndGroup();
    /// Ends the open step, so the next edit starts a new one.
    void Close() {
        open_ = false;
    }

    bool CanUndo() const {
        return next_ > 0;
    }
    bool CanRedo() const {
        return next_ < steps_.size();
    }
    /// @brief Undoes the last step or group; edits of prims removed since are skipped.
    /// @return Whether there was one.
    bool Undo(Scene& scene);
    bool Redo(Scene& scene);
    /// @return What `Undo` or `Redo` would do, as "/World/Cube xformOp:translate".
    std::string UndoLabel() const;
    std::string RedoLabel() const;
    /// Forgets every step, as when the viewer switches scenes.
    void Reset();
    /// @return The steps recorded, a group's edits each counted.
    std::size_t Size() const {
        return steps_.size();
    }

  private:
    /// One attribute's authored state: unset, a value, an array or time samples.
    struct Cell {
        std::optional<Value> value;
        Tensor array;
        std::vector<TimeSample> samples;
    };
    /// One prim of a subtree, enough to define it again.
    struct PrimSnapshot {
        Path path;
        std::string type;
        std::vector<std::string> apis;
        std::map<std::string, Cell> attributes;
    };
    struct Step {
        Path path;
        /// Attribute steps: each attribute's state before and after.
        std::vector<std::pair<std::string, std::pair<Cell, Cell>>> attributes;
        /// Structure steps: the subtree at `path` before and after, parents first; empty for none.
        std::vector<PrimSnapshot> before;
        std::vector<PrimSnapshot> after;
        bool structural = false;
        bool open = false;
        /// Steps of one non-zero group undo together.
        std::uint64_t group = 0;
    };

    static Cell Capture(const Scene& scene, PrimId prim, const std::string& attribute);
    static void Apply(Scene& scene, PrimId prim, const std::string& attribute, const Cell& cell);
    static std::vector<PrimSnapshot> Snapshot(const Scene& scene, const Path& path);
    static void Restore(Scene& scene, const std::vector<PrimSnapshot>& prims);
    static void Apply(Scene& scene, const Step& step, bool forward);
    void Push(Step step);
    bool Move(Scene& scene, bool forward);
    static std::string Label(const Step& step);

    std::vector<Step> steps_;
    /// Steps before it are done; from it on, undone.
    std::size_t next_ = 0;
    bool open_ = false;
    int groupDepth_ = 0;
    std::uint64_t group_ = 0;
    std::uint64_t nextGroup_ = 1;
};

}  // namespace ernie

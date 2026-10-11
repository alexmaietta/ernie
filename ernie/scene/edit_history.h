// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/value.h"
#include "ernie/util/tensor.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie {

/// Undo and redo for scene edits made through it: attribute sets and
/// clears, prims defined, and prims removed with their descendants.
/// Each records what it changes, so `Undo` puts it back (an
/// attribute's earlier value, or none; a removed subtree with its
/// attributes and tensors) and `Redo` does it again.
///
/// Continuous edits, such as dragging a slider or a gizmo, merge into
/// one step: an `open` edit of the same attribute as the last step
/// updates that step until `Close` (called when the drag ends). Edits
/// between `BeginGroup` and `EndGroup`, such as defining a prim and
/// setting its attributes, undo and redo together. Edits made to the
/// scene directly are not recorded; undoing past one sets the
/// attribute to what the history last saw.
///
/// Reference: E. Gamma, R. Helm, R. Johnson and J. Vlissides, "Design
/// Patterns", the Command pattern, Addison-Wesley, 1994.
class EditHistory {
  public:
    /// Sets `attribute` of the prim at `path` to `value` and records
    /// it, dropping what was undone (the redo steps). With `open`,
    /// merges into the last step if that is an open edit of the same
    /// attribute.
    /// @throws std::invalid_argument as `Scene::Set`.
    void Set(Scene& scene, const Path& path,
             std::string_view attribute, Value value,
             bool open = false);

    /// Sets several attributes of the prim at `path`, such as the
    /// transform ops of `LocalTransformEdits`, as one step. With
    /// `open`, merges into the last step if that is an open edit of
    /// the same attributes.
    /// @throws std::invalid_argument as `Scene::Set`.
    void Set(Scene& scene, const Path& path,
             std::vector<std::pair<std::string, Value>> edits,
             bool open = false);

    /// Clears `attribute` of the prim at `path` (back to its
    /// fallback) and records it.
    void Clear(Scene& scene, const Path& path,
               std::string_view attribute);

    /// Defines the prim at `path` with `type` and records it; undoing
    /// removes it.
    /// @throws std::invalid_argument as `Scene::Define`, or if a prim
    /// is there already.
    void Define(Scene& scene, const Path& path,
                std::string_view type);

    /// Removes the prim at `path` and its descendants and records
    /// them; undoing restores them, sharing their tensors. Returns
    /// whether there was a prim.
    bool Remove(Scene& scene, const Path& path);

    /// Copies the prim at `path` of `source` and its descendants,
    /// with their attributes and tensors (shared), to the same path
    /// of `scene` and records it; undoing removes them. For prims
    /// authored in a scratch scene first, such as a new camera or
    /// terrain.
    /// @throws std::invalid_argument if `source` has no prim at
    /// `path` or `scene` has one.
    void Add(Scene& scene, const Scene& source, const Path& path);

    /// Groups the edits until `EndGroup` into one step. Groups nest;
    /// the outermost counts.
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

    /// Undoes the last step (or group); returns whether there was
    /// one. Attributes of prims removed since are skipped.
    bool Undo(Scene& scene);

    /// Redoes the last undone step (or group); returns whether there
    /// was one.
    bool Redo(Scene& scene);

    /// A short description of the step `Undo` would undo, such as
    /// "/World/Cube xformOp:transform".
    std::string UndoLabel() const;
    std::string RedoLabel() const;

    /// Forgets every step, such as when the viewer switches scenes.
    void Reset();

    /// The number of steps recorded, counting a group's edits one by
    /// one.
    std::size_t Size() const {
        return steps_.size();
    }

  private:
    /// One prim of a removed subtree, enough to define it again.
    struct PrimSnapshot {
        Path path;
        std::string type;
        std::map<std::string, Value> values;
        std::map<std::string, Tensor> tensors;
    };

    struct Step {
        enum class Kind {
            Attribute,
            Define,
            Remove,
            Add
        } kind = Kind::Attribute;
        Path path;
        std::string attribute;  ///< Attribute steps.
        std::optional<Value>
            before;  ///< None when the attribute was not authored.
        std::optional<Value> after;  ///< None for a clear.
        std::string type;            ///< Define steps.
        std::vector<PrimSnapshot>
            prims;  ///< Remove and Add steps' subtree, parents first.
        std::uint64_t group =
            0;  ///< Steps sharing a non-zero group undo together.
    };

    void Push(Step step, bool open);
    void Record(Scene& scene, const Path& path,
                std::string_view attribute,
                std::optional<Value> value, bool open);
    /// Applies `step` forwards or backwards; false if it no longer
    /// applies (its prim is gone).
    static bool Apply(Scene& scene, const Step& step, bool forward);
    /// Undoes (`forward` false) or redoes the next group of steps;
    /// whether any applied.
    bool Move(Scene& scene, bool forward);
    static void Restore(Scene& scene,
                        const std::vector<PrimSnapshot>& prims);
    /// The prim at `path` of `scene` and its descendants, parents
    /// first.
    static std::vector<PrimSnapshot> Snapshot(const Scene& scene,
                                              const Path& path);
    std::string Label(const Step& step) const;

    std::vector<Step> steps_;
    std::size_t next_ =
        0;  ///< Steps before it are done; from it on, undone.
    bool open_ = false;
    int groupDepth_ = 0;
    std::uint64_t group_ =
        0;  ///< The open group's id, while `groupDepth_` > 0.
    std::uint64_t nextGroup_ = 1;
};

}  // namespace ernie

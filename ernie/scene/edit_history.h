#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/value.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// Undo and redo for attribute edits made through it: each `Set` records the attribute's value
/// before and after, so `Undo` sets it back (or clears it, if it had none) and `Redo` again.
///
/// Continuous edits, such as dragging a slider or a gizmo, merge into one step: an `open` edit
/// of the same attribute as the last step updates that step until `Close` (called when the drag
/// ends). Edits made to the scene directly are not recorded; undoing past one sets the attribute
/// to what the history last saw.
///
/// Reference: E. Gamma, R. Helm, R. Johnson and J. Vlissides, "Design Patterns", the Command
/// pattern, Addison-Wesley, 1994.
class EditHistory {
  public:
    /// Sets `attribute` of the prim at `path` to `value` and records it, dropping what was undone
    /// (the redo steps). With `open`, merges into the last step if that is an open edit of the
    /// same attribute.
    /// @throws std::invalid_argument as `Scene::Set`.
    void Set(Scene& scene, const Path& path, std::string_view attribute, Value value,
             bool open = false);

    /// Clears `attribute` of the prim at `path` (back to its fallback) and records it.
    void Clear(Scene& scene, const Path& path, std::string_view attribute);

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

    /// Undoes the last step; returns whether there was one. Prims removed since are skipped.
    bool Undo(Scene& scene);

    /// Redoes the last undone step; returns whether there was one.
    bool Redo(Scene& scene);

    /// A short description of the step `Undo` would undo, such as "/World/Cube xformOp:transform".
    std::string UndoLabel() const;
    std::string RedoLabel() const;

    /// Forgets every step, such as when the viewer switches scenes.
    void Reset();

    std::size_t Size() const {
        return steps_.size();
    }

  private:
    struct Step {
        Path path;
        std::string attribute;
        std::optional<Value> before;  ///< None when the attribute was not authored.
        std::optional<Value> after;   ///< None for a clear.
    };

    void Record(Scene& scene, const Path& path, std::string_view attribute,
                std::optional<Value> value, bool open);
    static void Apply(Scene& scene, const Step& step, const std::optional<Value>& value);

    std::vector<Step> steps_;
    std::size_t next_ = 0;  ///< Steps before it are done; from it on, undone.
    bool open_ = false;
};

}  // namespace ernie

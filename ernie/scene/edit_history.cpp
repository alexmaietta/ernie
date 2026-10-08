#include "ernie/scene/edit_history.h"

#include <utility>

namespace ernie {

void EditHistory::Set(Scene& scene, const Path& path, std::string_view attribute, Value value,
                      bool open) {
    Record(scene, path, attribute, std::move(value), open);
}

void EditHistory::Clear(Scene& scene, const Path& path, std::string_view attribute) {
    Record(scene, path, attribute, std::nullopt, false);
}

void EditHistory::Record(Scene& scene, const Path& path, std::string_view attribute,
                         std::optional<Value> value, bool open) {
    const Value* current = scene.Get(path, attribute);
    std::optional<Value> before;
    if (current != nullptr) {
        before = *current;
    }
    Step step{path, std::string(attribute), std::move(before), std::move(value)};
    Apply(scene, step, step.after);
    const bool merge = open && open_ && next_ == steps_.size() && next_ > 0 &&
                       steps_[next_ - 1].path == path && steps_[next_ - 1].attribute == attribute;
    if (merge) {
        steps_[next_ - 1].after = std::move(step.after);
        return;
    }
    steps_.resize(next_);
    steps_.push_back(std::move(step));
    next_ = steps_.size();
    open_ = open;
}

void EditHistory::Apply(Scene& scene, const Step& step, const std::optional<Value>& value) {
    if (value) {
        scene.Set(step.path, step.attribute, *value);
    } else {
        scene.Clear(step.path, step.attribute);
    }
}

bool EditHistory::Undo(Scene& scene) {
    open_ = false;
    while (next_ > 0) {
        const Step& step = steps_[--next_];
        if (scene.Has(step.path)) {
            Apply(scene, step, step.before);
            return true;
        }
    }
    return false;
}

bool EditHistory::Redo(Scene& scene) {
    open_ = false;
    while (next_ < steps_.size()) {
        const Step& step = steps_[next_++];
        if (scene.Has(step.path)) {
            Apply(scene, step, step.after);
            return true;
        }
    }
    return false;
}

std::string EditHistory::UndoLabel() const {
    return CanUndo() ? steps_[next_ - 1].path.String() + " " + steps_[next_ - 1].attribute : "";
}

std::string EditHistory::RedoLabel() const {
    return CanRedo() ? steps_[next_].path.String() + " " + steps_[next_].attribute : "";
}

void EditHistory::Reset() {
    steps_.clear();
    next_ = 0;
    open_ = false;
}

}  // namespace ernie

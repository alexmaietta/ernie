#include "ernie/scene/edit_history.h"

#include <stdexcept>
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
    Step step;
    step.path = path;
    step.attribute = std::string(attribute);
    if (current != nullptr) {
        step.before = *current;
    }
    step.after = std::move(value);
    Apply(scene, step, true);
    const bool merge = open && open_ && next_ == steps_.size() && next_ > 0 &&
                       steps_[next_ - 1].kind == Step::Kind::Attribute &&
                       steps_[next_ - 1].path == path && steps_[next_ - 1].attribute == attribute;
    if (merge) {
        steps_[next_ - 1].after = std::move(step.after);
        return;
    }
    Push(std::move(step), open);
}

void EditHistory::Define(Scene& scene, const Path& path, std::string_view type) {
    if (scene.Has(path)) {
        throw std::invalid_argument("A prim is at '" + path.String() + "' already");
    }
    Step step;
    step.kind = Step::Kind::Define;
    step.path = path;
    step.type = std::string(type);
    Apply(scene, step, true);
    Push(std::move(step), false);
}

bool EditHistory::Remove(Scene& scene, const Path& path) {
    if (path.IsEmpty() || path.IsRoot() || !scene.Has(path)) {
        return false;
    }
    Step step;
    step.kind = Step::Kind::Remove;
    step.path = path;
    step.prims = Snapshot(scene, path);
    Apply(scene, step, true);
    Push(std::move(step), false);
    return true;
}

void EditHistory::Add(Scene& scene, const Scene& source, const Path& path) {
    if (!source.Has(path)) {
        throw std::invalid_argument("No prim at '" + path.String() + "' to add");
    }
    if (scene.Has(path)) {
        throw std::invalid_argument("A prim is at '" + path.String() + "' already");
    }
    Step step;
    step.kind = Step::Kind::Add;
    step.path = path;
    step.prims = Snapshot(source, path);
    Apply(scene, step, true);
    Push(std::move(step), false);
}

std::vector<EditHistory::PrimSnapshot> EditHistory::Snapshot(const Scene& scene,
                                                             const Path& path) {
    std::vector<PrimSnapshot> prims;
    for (const Path& prim : scene.Prims()) {
        if (prim != path && !prim.HasPrefix(path)) {
            continue;
        }
        PrimSnapshot snapshot{prim, scene.TypeOf(prim), {}, {}};
        for (const std::string& attribute : scene.Attributes(prim)) {
            if (const Value* value = scene.Get(prim, attribute)) {
                snapshot.values.emplace(attribute, *value);
            } else if (const Tensor* tensor = scene.Read(prim, attribute)) {
                snapshot.tensors.emplace(attribute, *tensor);
            }
        }
        prims.push_back(std::move(snapshot));
    }
    return prims;
}

void EditHistory::Push(Step step, bool open) {
    step.group = groupDepth_ > 0 ? group_ : 0;
    steps_.resize(next_);
    steps_.push_back(std::move(step));
    next_ = steps_.size();
    open_ = open;
}

void EditHistory::BeginGroup() {
    if (groupDepth_++ == 0) {
        group_ = nextGroup_++;
        open_ = false;
    }
}

void EditHistory::EndGroup() {
    if (groupDepth_ > 0 && --groupDepth_ == 0) {
        group_ = 0;
        open_ = false;
    }
}

void EditHistory::Restore(Scene& scene, const std::vector<PrimSnapshot>& prims) {
    for (const PrimSnapshot& prim : prims) {
        // Untyped prims come back as the ancestors of typed ones, or as typeless prims here.
        scene.Define(prim.path, prim.type);
        for (const auto& [attribute, value] : prim.values) {
            scene.Set(prim.path, attribute, value);
        }
        for (const auto& [attribute, tensor] : prim.tensors) {
            scene.Write(prim.path, attribute, tensor);
        }
    }
}

bool EditHistory::Apply(Scene& scene, const Step& step, bool forward) {
    switch (step.kind) {
    case Step::Kind::Attribute: {
        if (!scene.Has(step.path)) {
            return false;
        }
        const std::optional<Value>& value = forward ? step.after : step.before;
        if (value) {
            scene.Set(step.path, step.attribute, *value);
        } else {
            scene.Clear(step.path, step.attribute);
        }
        return true;
    }
    case Step::Kind::Define:
        if (forward) {
            scene.Define(step.path, step.type);
            return true;
        }
        return scene.Remove(step.path);
    case Step::Kind::Remove:
    case Step::Kind::Add:
        // A removal goes forwards as an addition goes backwards.
        if (forward == (step.kind == Step::Kind::Remove)) {
            return scene.Remove(step.path);
        }
        Restore(scene, step.prims);
        return true;
    }
    return false;
}

bool EditHistory::Move(Scene& scene, bool forward) {
    bool applied = false;
    if (forward ? next_ == steps_.size() : next_ == 0) {
        return false;
    }
    const std::uint64_t group = steps_[forward ? next_ : next_ - 1].group;
    do {
        applied |= forward ? Apply(scene, steps_[next_++], true)
                           : Apply(scene, steps_[--next_], false);
    } while (group != 0 &&
             (forward ? next_ < steps_.size() && steps_[next_].group == group
                      : next_ > 0 && steps_[next_ - 1].group == group));
    return applied;
}

bool EditHistory::Undo(Scene& scene) {
    open_ = false;
    // Steps whose prims are gone no longer apply; skip to one that does.
    while (CanUndo()) {
        if (Move(scene, false)) {
            return true;
        }
    }
    return false;
}

bool EditHistory::Redo(Scene& scene) {
    open_ = false;
    while (CanRedo()) {
        if (Move(scene, true)) {
            return true;
        }
    }
    return false;
}

std::string EditHistory::Label(const Step& step) const {
    switch (step.kind) {
    case Step::Kind::Define:
    case Step::Kind::Add:
        return "Create " + step.path.String();
    case Step::Kind::Remove:
        return "Delete " + step.path.String();
    case Step::Kind::Attribute:
        break;
    }
    return step.path.String() + " " + step.attribute;
}

std::string EditHistory::UndoLabel() const {
    if (!CanUndo()) {
        return "";
    }
    // A group reads as its first step, such as the prim it created.
    std::size_t first = next_ - 1;
    while (steps_[first].group != 0 && first > 0 && steps_[first - 1].group == steps_[first].group) {
        --first;
    }
    return Label(steps_[first]);
}

std::string EditHistory::RedoLabel() const {
    return CanRedo() ? Label(steps_[next_]) : "";
}

void EditHistory::Reset() {
    steps_.clear();
    next_ = 0;
    open_ = false;
}

}  // namespace ernie

// SPDX-License-Identifier: BSD-3-Clause
/// @file edit_history.cpp
/// @brief Recording edits as steps of before and after states, and applying them either way.
#include "ernie/scene/edit_history.h"

#include "ernie/util/text.h"

#include <algorithm>
#include <stdexcept>

namespace ernie {

void EditHistory::Set(Scene& scene, const Path& path,
                      std::vector<std::pair<std::string, Value>> edits, bool open) {
    const PrimId prim = scene.Require(path);
    Step step{.path = path, .open = open};
    try {
        for (auto& [name, value] : edits) {
            Cell before = Capture(scene, prim, name);
            scene.Set(prim, name, std::move(value));
            step.attributes.push_back({name, std::move(before), Capture(scene, prim, name)});
        }
    } catch (...) {
        // What applied before the rejected edit stays undoable.
        if (!step.attributes.empty()) {
            Push(std::move(step));
        }
        throw;
    }
    const auto same = [&](const Step& last) {
        return !last.structural && last.open && last.path == path &&
               std::ranges::equal(last.attributes, step.attributes, {}, &AttributeEdit::name,
                                  &AttributeEdit::name);
    };
    if (open && open_ && next_ == steps_.size() && !steps_.empty() && same(steps_.back())) {
        for (std::size_t i = 0; i < step.attributes.size(); ++i) {
            steps_.back().attributes[i].after = std::move(step.attributes[i].after);
        }
        return;
    }
    Push(std::move(step));
}

void EditHistory::Clear(Scene& scene, const Path& path, std::string_view attribute) {
    const PrimId prim = scene.Require(path);
    const std::string name(attribute);
    Cell before = Capture(scene, prim, name);
    scene.Clear(prim, attribute);
    Push({.path = path, .attributes = {{name, std::move(before), Capture(scene, prim, name)}}});
}

void EditHistory::Define(Scene& scene, const Path& path, std::string_view type) {
    if (scene.Has(path)) {
        throw std::invalid_argument("'" + path.String() + "' has a prim already");
    }
    scene.Define(path, type);
    Push({.path = path, .after = Snapshot(scene, path), .structural = true});
}

bool EditHistory::Remove(Scene& scene, const Path& path) {
    if (!scene.Has(path)) {
        return false;
    }
    Step step{.path = path, .before = Snapshot(scene, path), .structural = true};
    scene.Remove(path);
    Push(std::move(step));
    return true;
}

void EditHistory::Add(Scene& scene, const Scene& source, const Path& path) {
    if (!source.Has(path) || scene.Has(path)) {
        throw std::invalid_argument("Cannot add '" + path.String() +
                                    "': the source has no prim there or the scene has one");
    }
    Step step{.path = path, .after = Snapshot(source, path), .structural = true};
    Restore(scene, step.after);
    Push(std::move(step));
}

void EditHistory::Copy(Scene& scene, const Path& from, const Path& to) {
    if (!scene.Has(from) || scene.Has(to) || to.HasPrefix(from)) {
        throw std::invalid_argument("Cannot copy '" + from.String() + "' to '" + to.String() +
                                    "': no prim there, one at the copy, or the copy inside it");
    }
    std::vector<PrimSnapshot> prims = Snapshot(scene, from);
    const std::size_t prefix = from.String().size();
    for (PrimSnapshot& prim : prims) {
        prim.path = Path(to.String() + prim.path.String().substr(prefix));
    }
    Step step{.path = to, .after = std::move(prims), .structural = true};
    Restore(scene, step.after);
    Push(std::move(step));
}

void EditHistory::BeginGroup() {
    if (groupDepth_++ == 0) {
        group_ = nextGroup_++;
    }
}

void EditHistory::EndGroup() {
    if (groupDepth_ > 0 && --groupDepth_ == 0) {
        group_ = 0;
    }
}

bool EditHistory::Undo(Scene& scene) {
    return Move(scene, false);
}

bool EditHistory::Redo(Scene& scene) {
    return Move(scene, true);
}

std::string EditHistory::UndoLabel() const {
    return CanUndo() ? Label(steps_[next_ - 1]) : std::string();
}

std::string EditHistory::RedoLabel() const {
    return CanRedo() ? Label(steps_[next_]) : std::string();
}

void EditHistory::Reset() {
    *this = EditHistory();
}

EditHistory::Cell EditHistory::Capture(const Scene& scene, PrimId prim,
                                       const std::string& attribute) {
    Cell cell;
    const std::span<const TimeSample> samples = scene.Samples(prim, attribute);
    cell.samples.assign(samples.begin(), samples.end());
    if (const Tensor* array = scene.Read(prim, attribute)) {
        cell.array = *array;
    } else {
        cell.value = scene.Get(prim, attribute);
    }
    return cell;
}

void EditHistory::Apply(Scene& scene, PrimId prim, const std::string& attribute, const Cell& cell) {
    if (!cell.samples.empty()) {
        scene.ClearSamples(prim, attribute);
        for (const TimeSample& sample : cell.samples) {
            sample.tensor.Empty() ? scene.SetAt(prim, attribute, sample.time, sample.value)
                                  : scene.WriteAt(prim, attribute, sample.time, sample.tensor);
        }
    } else if (!cell.array.Empty()) {
        scene.Write(prim, attribute, cell.array);
    } else if (cell.value) {
        scene.Set(prim, attribute, *cell.value);
    } else {
        scene.Clear(prim, attribute);
    }
}

std::vector<EditHistory::PrimSnapshot> EditHistory::Snapshot(const Scene& scene, const Path& path) {
    std::vector<PrimSnapshot> prims;
    std::vector<Path> pending{path};
    while (!pending.empty()) {
        const Path next = std::move(pending.back());
        pending.pop_back();
        const PrimId prim = scene.Require(next);
        PrimSnapshot& snapshot = prims.emplace_back();
        snapshot.path = next;
        snapshot.type = std::string(scene.TypeName(prim));
        snapshot.apis = scene.AppliedApis(prim);
        for (const std::string& name : scene.Attributes(next)) {
            snapshot.attributes.emplace(name, Capture(scene, prim, name));
        }
        const std::vector<Path> children = scene.Children(next);
        pending.insert(pending.end(), children.rbegin(), children.rend());
    }
    return prims;
}

void EditHistory::Restore(Scene& scene, const std::vector<PrimSnapshot>& prims) {
    for (const PrimSnapshot& snapshot : prims) {
        const PrimId prim = scene.Define(snapshot.path, snapshot.type);
        for (const std::string& api : snapshot.apis) {
            scene.ApplyApi(prim, api);
        }
        for (const auto& [name, cell] : snapshot.attributes) {
            Apply(scene, prim, name, cell);
        }
    }
}

void EditHistory::Apply(Scene& scene, const Step& step, bool forward) {
    if (step.structural) {
        scene.Remove(step.path);
        Restore(scene, forward ? step.after : step.before);
        return;
    }
    // Edits of a prim removed since are skipped.
    const std::optional<PrimId> prim = scene.Find(step.path);
    if (!prim) {
        return;
    }
    for (std::size_t i = 0; i < step.attributes.size(); ++i) {
        const AttributeEdit& edit = step.attributes[forward ? i : step.attributes.size() - 1 - i];
        Apply(scene, *prim, edit.name, forward ? edit.after : edit.before);
    }
}

void EditHistory::Push(Step step) {
    steps_.resize(next_);
    step.group = groupDepth_ > 0 ? group_ : 0;
    open_ = step.open;
    steps_.push_back(std::move(step));
    next_ = steps_.size();
}

bool EditHistory::Move(Scene& scene, bool forward) {
    open_ = false;
    if (forward ? !CanRedo() : !CanUndo()) {
        return false;
    }
    const std::uint64_t group = steps_[forward ? next_ : next_ - 1].group;
    do {
        if (forward) {
            Apply(scene, steps_[next_++], true);
        } else {
            Apply(scene, steps_[--next_], false);
        }
    } while (group != 0 && (forward ? CanRedo() && steps_[next_].group == group
                                    : CanUndo() && steps_[next_ - 1].group == group));
    return true;
}

std::string EditHistory::Label(const Step& step) {
    if (step.structural) {
        return (step.before.empty()  ? "Add "
                : step.after.empty() ? "Remove "
                                     : "Change ") +
               step.path.String();
    }
    return step.path.String() + " " + Join(step.attributes, " ", &AttributeEdit::name);
}

}  // namespace ernie

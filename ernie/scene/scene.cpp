// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/scene.h"

#include "ernie/scene/prim_type.h"

#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <set>
#include <stdexcept>
#include <utility>

namespace ernie {

namespace {

std::size_t StampIndex(Dirty bit) {
    return static_cast<std::size_t>(std::countr_zero(static_cast<std::uint32_t>(bit)));
}

Dirty StampBit(std::size_t index) {
    return static_cast<Dirty>(1u << index);
}

std::invalid_argument NoPrim(const Path& path) {
    return std::invalid_argument("No prim at '" + path.String() + "'");
}

/// `relative` ("Rock", or "Rig/Arm") below `path`.
Path Below(Path path, std::string_view relative) {
    std::size_t start = 0;
    while (true) {
        const std::size_t slash = relative.find('/', start);
        path = path.Child(relative.substr(start, slash - start));
        if (slash == std::string_view::npos) {
            return path;
        }
        start = slash + 1;
    }
}

}  // namespace

Dirty DirtyOf(std::string_view attribute) {
    if (IsXformAttribute(attribute)) {
        return Dirty::Transform;
    }
    // What the registered types' declarations say a change to each attribute stamps
    // (`AttributeMeta::dirty`).
    static const std::map<std::string_view, Dirty, std::less<>> declared = [] {
        std::map<std::string_view, Dirty, std::less<>> bits;
        for (const PrimTypeDef* type : PrimTypes()) {
            for (const AttributeDef& def : type->attributes) {
                if (def.meta.dirty) {
                    bits.emplace(def.name, *def.meta.dirty);
                }
            }
        }
        return bits;
    }();
    if (const auto it = declared.find(attribute); it != declared.end()) {
        return it->second;
    }
    if (attribute.starts_with("primvars:")) {
        return Dirty::Primvars;
    }
    return Dirty::Params;
}

bool Prim::IsValid() const {
    return scene_ != nullptr && scene_->Has(path_);
}

const std::string& Prim::Type() const {
    if (scene_ == nullptr) {
        throw NoPrim(path_);
    }
    return scene_->TypeOf(path_);
}

Prim Prim::Define(std::string_view relative, std::string_view type) const {
    if (scene_ == nullptr) {
        throw NoPrim(path_);
    }
    return scene_->Define(Below(path_, relative), type);
}

Prim Prim::GetPrim(std::string_view relative) const {
    return scene_ != nullptr ? scene_->GetPrim(Below(path_, relative)) : Prim();
}

Prim& Prim::Set(std::string_view attribute, Value value) {
    if (scene_ == nullptr) {
        throw NoPrim(path_);
    }
    scene_->Set(path_, attribute, std::move(value));
    return *this;
}

Prim& Prim::SetXform(const XformOps& xform) {
    if (scene_ == nullptr) {
        throw NoPrim(path_);
    }
    ernie::SetXform(*scene_, path_, xform);
    return *this;
}

Prim& Prim::Write(std::string_view attribute, Tensor tensor) {
    if (scene_ == nullptr) {
        throw NoPrim(path_);
    }
    scene_->Write(path_, attribute, std::move(tensor));
    return *this;
}

const Value* Prim::Get(std::string_view attribute) const {
    return scene_ != nullptr ? scene_->Get(path_, attribute) : nullptr;
}

const Tensor* Prim::Read(std::string_view attribute) const {
    return scene_ != nullptr ? scene_->Read(path_, attribute) : nullptr;
}

Scene::UniqueId::UniqueId() {
    static std::atomic<std::uint64_t> next{1};
    value = next.fetch_add(1, std::memory_order_relaxed);
}

Scene::Scene() {
    prims_.emplace(Path::Root(), PrimData{});
}

Scene::PrimData& Scene::Find(const Path& path) {
    const auto it = prims_.find(path);
    if (it == prims_.end()) {
        throw NoPrim(path);
    }
    return it->second;
}

const Scene::PrimData& Scene::Find(const Path& path) const {
    const auto it = prims_.find(path);
    if (it == prims_.end()) {
        throw NoPrim(path);
    }
    return it->second;
}

void Scene::Stamp(const Path& path, PrimData& prim, Dirty bit) {
    if (bit == Dirty::None) {
        return;  // nothing to record, and no stamp of its own
    }
    const std::size_t index = StampIndex(bit);
    // Moved to its new place in `stamped_`, for `Changes`.
    const auto restamp = [&](const Path& stamped, PrimData& data) {
        data.stamps[index] = version_;
        if (data.latest != version_) {
            stamped_.erase({data.latest, stamped});
            stamped_.emplace(version_, stamped);
            data.latest = version_;
        }
    };
    restamp(path, prim);
    if (bit == Dirty::Transform || bit == Dirty::Visibility) {
        for (auto it = prims_.upper_bound(path); it != prims_.end() && it->first.HasPrefix(path);
             ++it) {
            restamp(it->first, it->second);
        }
    }
}

void Scene::Stamp(const Path& path, PrimData& prim, std::string_view attribute) {
    if (const auto it = prim.versions.find(attribute); it != prim.versions.end()) {
        it->second = version_;
    } else {
        prim.versions.emplace(std::string(attribute), version_);
    }
    if (IsXformAttribute(attribute)) {
        prim.ownTransform = version_;
    }
    Stamp(path, prim, DirtyOf(attribute));
}

Scene::PrimData& Scene::CreateIfMissing(const Path& path) {
    const auto it = prims_.find(path);
    if (it != prims_.end()) {
        return it->second;
    }
    CreateIfMissing(path.Parent());
    PrimData& prim = prims_[path];
    Stamp(path, prim, Dirty::Added);
    return prim;
}

Prim Scene::Define(const Path& path, std::string_view type) {
    if (path.IsEmpty() || path.IsRoot()) {
        throw std::invalid_argument("Cannot define the prim at '" + path.String() + "'");
    }
    ++version_;
    const bool existed = Has(path);
    PrimData& prim = CreateIfMissing(path);
    if (prim.type != type) {
        if (existed) {
            prim.values.clear();
            prim.tensors.clear();
            prim.versions.clear();
            removed_[path] = version_;
            Stamp(path, prim, Dirty::Added);
        }
        if (!prim.type.empty()) {
            byType_[prim.type].erase(path);
        }
        if (!type.empty()) {
            byType_[std::string(type)].insert(path);
        }
        prim.type = type;
        prim.schema = FindPrimType(type);
    }
    return Prim(this, path);
}

Prim Scene::Clone(const Path& source, const Path& destination) {
    if (!Has(source)) {
        throw std::invalid_argument("Cannot clone '" + source.String() + "', which does not exist");
    }
    if (destination.IsEmpty() || destination.IsRoot() || Has(destination) ||
        destination.HasPrefix(source)) {
        throw std::invalid_argument("Cannot clone '" + source.String() + "' to '" +
                                    destination.String() +
                                    "': the destination must be a new prim outside the source");
    }
    // Paths inside the source move with it.
    const auto remap = [&](const Path& path) {
        if (!path.HasPrefix(source)) {
            return path;
        }
        return Path(destination.String() + path.String().substr(source.String().size()));
    };
    std::vector<Path> paths{source};
    for (std::size_t i = 0; i < paths.size(); ++i) {
        const std::vector<Path> children = Children(paths[i]);
        paths.insert(paths.end(), children.begin(), children.end());
    }
    for (const Path& path : paths) {
        const Path copy = remap(path);
        Define(copy, TypeOf(path));
        for (const std::string& attribute : Attributes(path)) {
            if (const Tensor* tensor = Read(path, attribute)) {
                Write(copy, attribute, *tensor, WriteStream(path, attribute));
                continue;
            }
            const Value* value = Get(path, attribute);
            if (value == nullptr) {
                continue;
            }
            if (const auto* target = value->Get<Path>()) {
                Set(copy, attribute, remap(*target));
            } else if (const auto* targets = value->Get<std::vector<Path>>()) {
                std::vector<Path> moved;
                for (const Path& each : *targets) {
                    moved.push_back(remap(each));
                }
                Set(copy, attribute, std::move(moved));
            } else {
                Set(copy, attribute, *value);
            }
        }
    }
    return GetPrim(destination);
}

bool Scene::Remove(const Path& path) {
    if (path.IsRoot()) {
        throw std::invalid_argument("Cannot remove the root prim");
    }
    auto first = prims_.find(path);
    if (first == prims_.end()) {
        return false;
    }
    ++version_;
    auto last = first;
    while (last != prims_.end() && last->first.HasPrefix(path)) {
        removed_[last->first] = version_;
        stamped_.erase({last->second.latest, last->first});
        if (const auto typed = byType_.find(last->second.type); typed != byType_.end()) {
            typed->second.erase(last->first);
        }
        ++last;
    }
    prims_.erase(first, last);
    ++removals_;
    return true;
}

bool Scene::Has(const Path& path) const {
    return prims_.contains(path);
}

Prim Scene::GetPrim(const Path& path) {
    return Has(path) ? Prim(this, path) : Prim();
}

const std::string& Scene::TypeOf(const Path& path) const {
    return Find(path).type;
}

std::vector<Path> Scene::Children(const Path& path) const {
    std::vector<Path> children;
    for (auto it = prims_.upper_bound(path); it != prims_.end() && it->first.HasPrefix(path);
         ++it) {
        if (it->first.Parent() == path) {
            children.push_back(it->first);
        }
    }
    return children;
}

std::vector<Path> Scene::Prims(std::string_view type) const {
    if (!type.empty()) {
        // By the index of each type's prims, in path order as the walk below gives them.
        const auto typed = byType_.find(type);
        return typed == byType_.end()
                   ? std::vector<Path>{}
                   : std::vector<Path>(typed->second.begin(), typed->second.end());
    }
    std::vector<Path> paths;
    for (const auto& [path, prim] : prims_) {
        if (!path.IsRoot() && (type.empty() || prim.type == type)) {
            paths.push_back(path);
        }
    }
    return paths;
}

void Scene::Set(const Path& path, std::string_view attribute, Value value) {
    PrimData& prim = Find(path);
    Validate(prim.schema, path, attribute, &value);
    ++version_;
    if (const auto tensor = prim.tensors.find(attribute); tensor != prim.tensors.end()) {
        prim.tensors.erase(tensor);
    }
    if (const auto row = prim.rows.find(attribute); row != prim.rows.end()) {
        prim.rows.erase(row);
    }
    if (const auto it = prim.values.find(attribute); it != prim.values.end()) {
        it->second = std::move(value);
    } else {
        prim.values.emplace(std::string(attribute), std::move(value));
    }
    Stamp(path, prim, attribute);
}

const Value* Scene::Get(const Path& path, std::string_view attribute) const {
    const auto prim = prims_.find(path);
    if (prim == prims_.end()) {
        return nullptr;
    }
    const auto it = prim->second.values.find(attribute);
    return it != prim->second.values.end() ? &it->second : nullptr;
}

void Scene::Write(const Path& path, std::string_view attribute, Tensor tensor,
                  std::uintptr_t stream) {
    WriteTensor(path, Find(path), attribute, std::move(tensor), stream);
}

void Scene::WriteTensor(const Path& path, PrimData& prim, std::string_view attribute, Tensor tensor,
                        std::uintptr_t stream, std::optional<std::pair<Tensor, std::int64_t>> row,
                        bool checked) {
    if (!checked) {
        Validate(prim.schema, path, attribute, nullptr);
    }
    // Found before inserted, so rewriting an attribute every step allocates no keys.
    if (const auto it = prim.streams.find(attribute); it != prim.streams.end()) {
        it->second = stream;
    } else {
        prim.streams.emplace(std::string(attribute), stream);
    }
    if (const auto it = prim.rows.find(attribute); it != prim.rows.end()) {
        if (row) {
            it->second = std::move(*row);
        } else {
            prim.rows.erase(it);
        }
    } else if (row) {
        prim.rows.emplace(std::string(attribute), std::move(*row));
    }
    ++version_;
    if (const auto value = prim.values.find(attribute); value != prim.values.end()) {
        prim.values.erase(value);
    }
    if (const auto it = prim.tensors.find(attribute); it != prim.tensors.end()) {
        it->second = std::move(tensor);
    } else {
        prim.tensors.emplace(std::string(attribute), std::move(tensor));
    }
    Stamp(path, prim, attribute);
}

Tensor Scene::Map(const Path& path, std::string_view attribute) {
    const PrimData& prim = Find(path);
    const auto it = prim.tensors.find(attribute);
    if (it == prim.tensors.end()) {
        throw std::invalid_argument("Cannot map " + std::string(attribute) + " of '" +
                                    path.String() + "', which is not a written array");
    }
    return it->second;
}

void Scene::Unmap(const Path& path, std::string_view attribute, std::uintptr_t stream) {
    PrimData& prim = Find(path);
    if (!prim.tensors.contains(attribute)) {
        throw std::invalid_argument("Cannot unmap " + std::string(attribute) + " of '" +
                                    path.String() + "', which is not a written array");
    }
    ++version_;
    prim.streams[std::string(attribute)] = stream;
    Stamp(path, prim, attribute);
}

void Scene::Unmap(const PrimSet& prims, std::string_view attribute, std::uintptr_t stream) {
    for (const Path& path : prims.Paths()) {
        if (!Find(path).tensors.contains(attribute)) {
            throw std::invalid_argument("Cannot unmap " + std::string(attribute) + " of '" +
                                        path.String() + "', which is not a written array");
        }
    }
    for (const Path& path : prims.Paths()) {
        Unmap(path, attribute, stream);
    }
}

std::uintptr_t Scene::WriteStream(const Path& path, std::string_view attribute) const {
    if (!Has(path)) {
        return 1;
    }
    const PrimData& prim = Find(path);
    const auto it = prim.streams.find(attribute);
    return it != prim.streams.end() ? it->second : 1;
}

void Scene::Write(std::span<const Path> paths, std::string_view attribute, const Tensor& rows,
                  std::uintptr_t stream) {
    Write(PrimSet(std::vector<Path>(paths.begin(), paths.end())), attribute, rows, stream);
}

void Scene::Write(const PrimSet& prims, std::string_view attribute, const Tensor& rows,
                  std::uintptr_t stream) {
    if (rows.Rank() == 0 || rows.Shape()[0] != static_cast<std::int64_t>(prims.Size())) {
        throw std::invalid_argument("Scene::Write needs one tensor row per path");
    }
    if (prims.sceneId_ != Id() || prims.removals_ != removals_ ||
        prims.prims_.size() != prims.Size()) {
        // Looked up all first, so a missing prim writes nothing.
        std::vector<void*> found;
        found.reserve(prims.Size());
        for (const Path& path : prims.Paths()) {
            found.push_back(&Find(path));
        }
        prims.prims_ = std::move(found);
        prims.sceneId_ = Id();
        prims.removals_ = removals_;
    }
    // The attribute is checked once per type among the prims (`Validate`), not once per prim,
    // and all before any is written, so a rejected one writes nothing.
    std::vector<const PrimTypeDef*> checked;
    for (std::size_t i = 0; i < prims.Size(); ++i) {
        const PrimTypeDef* schema = static_cast<PrimData*>(prims.prims_[i])->schema;
        if (std::find(checked.begin(), checked.end(), schema) == checked.end()) {
            Validate(schema, prims.Paths()[i], attribute, nullptr);
            checked.push_back(schema);
        }
    }
    for (std::size_t i = 0; i < prims.Size(); ++i) {
        const auto row = static_cast<std::int64_t>(i);
        WriteTensor(prims.Paths()[i], *static_cast<PrimData*>(prims.prims_[i]), attribute,
                    rows.Row(row), stream, std::pair(rows, row), true);
    }
}

const std::pair<Tensor, std::int64_t>* Scene::RowOf(const Path& path,
                                                    std::string_view attribute) const {
    const auto prim = prims_.find(path);
    if (prim == prims_.end()) {
        return nullptr;
    }
    const auto it = prim->second.rows.find(attribute);
    return it != prim->second.rows.end() ? &it->second : nullptr;
}

std::uint64_t Scene::OwnTransformVersion(const Path& path) const {
    const auto prim = prims_.find(path);
    return prim != prims_.end() ? prim->second.ownTransform : 0;
}

const Tensor* Scene::Read(const Path& path, std::string_view attribute) const {
    const auto prim = prims_.find(path);
    if (prim == prims_.end()) {
        return nullptr;
    }
    const auto it = prim->second.tensors.find(attribute);
    return it != prim->second.tensors.end() ? &it->second : nullptr;
}

std::vector<std::string> Scene::Attributes(const Path& path) const {
    const PrimData& prim = Find(path);
    std::set<std::string> names;
    for (const auto& [name, value] : prim.values) {
        names.insert(name);
    }
    for (const auto& [name, tensor] : prim.tensors) {
        names.insert(name);
    }
    return {names.begin(), names.end()};
}

bool Scene::Clear(const Path& path, std::string_view attribute) {
    PrimData& prim = Find(path);
    const auto value = prim.values.find(attribute);
    const auto tensor = prim.tensors.find(attribute);
    if (value == prim.values.end() && tensor == prim.tensors.end()) {
        return false;
    }
    ++version_;
    if (value != prim.values.end()) {
        prim.values.erase(value);
    }
    if (tensor != prim.tensors.end()) {
        prim.tensors.erase(tensor);
    }
    if (const auto row = prim.rows.find(attribute); row != prim.rows.end()) {
        prim.rows.erase(row);
    }
    Stamp(path, prim, attribute);
    return true;
}

std::uint64_t Scene::AttributeVersion(const Path& path, std::string_view attribute) const {
    const auto prim = prims_.find(path);
    if (prim == prims_.end()) {
        return 0;
    }
    const auto it = prim->second.versions.find(attribute);
    return it != prim->second.versions.end() ? it->second : 0;
}

glm::dmat4 Scene::LocalTransform(const Path& path) const {
    Find(path);
    return ComposeXformOps(*this, path);
}

glm::dmat4 Scene::WorldTransform(const Path& path) const {
    glm::dmat4 world = LocalTransform(path);
    if (ResetsXformStack(*this, path)) {
        return world;
    }
    for (Path parent = path.Parent(); !parent.IsEmpty(); parent = parent.Parent()) {
        world = LocalTransform(parent) * world;
        if (ResetsXformStack(*this, parent)) {
            break;
        }
    }
    return world;
}

void Scene::SetLocalTransform(const Path& path, const glm::dmat4& local) {
    for (auto& [attribute, value] : LocalTransformEdits(*this, path, local)) {
        Set(path, attribute, std::move(value));
    }
}

bool Scene::IsVisible(const Path& path) const {
    Find(path);
    for (Path p = path; !p.IsEmpty(); p = p.Parent()) {
        const Value* visibility = Get(p, tokens::kVisibility);
        const std::string* token = visibility != nullptr ? visibility->Get<std::string>() : nullptr;
        if (token != nullptr && *token == tokens::kInvisible) {
            return false;
        }
    }
    return true;
}

std::vector<Change> Scene::Changes(ChangeCursor& cursor) const {
    std::map<Path, Dirty> changed;
    for (const auto& [path, when] : removed_) {
        if (when > cursor.version) {
            changed[path] = Dirty::Removed;
        }
    }
    // Only the prims stamped since the cursor, by `stamped_`.
    for (auto it = stamped_.lower_bound({cursor.version + 1, Path()}); it != stamped_.end(); ++it) {
        const Path& path = it->second;
        if (path.IsRoot()) {
            continue;
        }
        const PrimData& prim = prims_.at(path);
        Dirty bits = Dirty::None;
        if (prim.stamps[StampIndex(Dirty::Added)] > cursor.version) {
            bits = Dirty::Added;
        } else {
            for (std::size_t i = 0; i < kStampCount; ++i) {
                if (prim.stamps[i] > cursor.version) {
                    bits |= StampBit(i);
                }
            }
        }
        if (bits != Dirty::None) {
            changed[path] |= bits;
        }
    }
    cursor.version = version_;
    std::vector<Change> changes;
    changes.reserve(changed.size());
    for (auto& [path, bits] : changed) {
        changes.push_back({path, bits});
    }
    return changes;
}

Overlay& Scene::GetOverlay(std::string_view name) {
    const auto found = overlays_.find(name);
    if (found != overlays_.end()) {
        return found->second;
    }
    return overlays_.emplace(std::string(name), Overlay()).first->second;
}

const Overlay* Scene::FindOverlay(std::string_view name) const {
    const auto found = overlays_.find(name);
    return found != overlays_.end() ? &found->second : nullptr;
}

bool Scene::RemoveOverlay(std::string_view name) {
    const auto found = overlays_.find(name);
    if (found == overlays_.end()) {
        return false;
    }
    overlays_.erase(found);
    return true;
}

std::vector<std::string> Scene::OverlayNames() const {
    std::vector<std::string> names;
    for (const auto& [name, overlay] : overlays_) {
        names.push_back(name);
    }
    return names;
}

}  // namespace ernie

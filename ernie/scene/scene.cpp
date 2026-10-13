// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/scene.h"

#include "ernie/scene/prim_type.h"
#include "ernie/scene/stage_schemas.h"

#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "ernie/util/profile.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <set>
#include <stdexcept>
#include <utility>

namespace ernie {

namespace {

std::size_t StampIndex(Dirty bit) {
    return static_cast<std::size_t>(
        std::countr_zero(static_cast<std::uint32_t>(bit)));
}

Dirty StampBit(std::size_t index) {
    return static_cast<Dirty>(1u << index);
}

std::invalid_argument NoPrim(const Path& path) {
    return std::invalid_argument("No prim at '" + path.String() +
                                 "'");
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
    // What the registered types' declarations say a change to each
    // attribute stamps (`AttributeMeta::dirty`).
    static const std::map<std::string_view, Dirty, std::less<>>
        declared = [] {
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
    if (const auto it = declared.find(attribute);
        it != declared.end()) {
        return it->second;
    }
    if (attribute.starts_with("primvars:")) {
        return Dirty::Primvars;
    }
    return Dirty::Params;
}

namespace {

/// The declared attributes that apply to descendants too
/// (`AttributeMeta::inherited`).
bool IsInherited(std::string_view attribute) {
    static const std::set<std::string_view, std::less<>> inherited = [] {
        std::set<std::string_view, std::less<>> names;
        for (const PrimTypeDef* type : PrimTypes()) {
            for (const AttributeDef& def : type->attributes) {
                if (def.meta.inherited) {
                    names.insert(def.name);
                }
            }
        }
        return names;
    }();
    return inherited.contains(attribute);
}

/// The declared relationships whose targets' changes reach the prims
/// holding them, with what they stamp (`AttributeMeta::targetDirty`).
const std::map<std::string_view, Dirty, std::less<>>& TargetDirties() {
    static const std::map<std::string_view, Dirty, std::less<>> bits = [] {
        std::map<std::string_view, Dirty, std::less<>> declared;
        for (const PrimTypeDef* type : PrimTypes()) {
            for (const AttributeDef& def : type->attributes) {
                if (def.meta.targetDirty) {
                    declared[def.name] |= *def.meta.targetDirty;
                }
            }
        }
        return declared;
    }();
    return bits;
}

}  // namespace

bool Prim::IsValid() const {
    return scene_ != nullptr && scene_->Has(path_);
}

const std::string& Prim::Type() const {
    if (scene_ == nullptr) {
        throw NoPrim(path_);
    }
    return scene_->TypeOf(path_);
}

Prim Prim::Define(std::string_view relative,
                  std::string_view type) const {
    if (scene_ == nullptr) {
        throw NoPrim(path_);
    }
    return scene_->Define(Below(path_, relative), type);
}

Prim Prim::GetPrim(std::string_view relative) const {
    return scene_ != nullptr ? scene_->GetPrim(Below(path_, relative))
                             : Prim();
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

Prim& Prim::SetAt(std::string_view attribute, double time,
                  Value value) {
    if (scene_ == nullptr) {
        throw NoPrim(path_);
    }
    scene_->SetAt(path_, attribute, time, std::move(value));
    return *this;
}

Prim& Prim::WriteAt(std::string_view attribute, double time,
                    Tensor tensor) {
    if (scene_ == nullptr) {
        throw NoPrim(path_);
    }
    scene_->WriteAt(path_, attribute, time, std::move(tensor));
    return *this;
}

const Value* Prim::Get(std::string_view attribute) const {
    return scene_ != nullptr ? scene_->Get(path_, attribute)
                             : nullptr;
}

const Tensor* Prim::Read(std::string_view attribute) const {
    return scene_ != nullptr ? scene_->Read(path_, attribute)
                             : nullptr;
}

std::vector<std::string> Prim::Attributes() const {
    return scene_ != nullptr ? scene_->Attributes(path_)
                             : std::vector<std::string>{};
}

void Scene::IdentifyPrimSet(const PrimSet& prims) {
    static std::atomic<std::uint64_t> next{1};
    if (prims.id_ == 0) {
        prims.id_ = next.fetch_add(1, std::memory_order_relaxed);
    }
}

Scene::UniqueId::UniqueId() {
    static std::atomic<std::uint64_t> next{1};
    value = next.fetch_add(1, std::memory_order_relaxed);
}

Scene::Scene() {
    // The root holds the stage metadata (`Stage`).
    PrimData root;
    root.schema = &Stage::Definition();
    prims_.emplace(Path::Root(), std::move(root));
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

void Scene::Stamp(const Path& path, PrimData& prim, Dirty bit,
                  bool descendants) {
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
    if (descendants || bit == Dirty::Transform ||
        bit == Dirty::Visibility) {
        for (auto it = prims_.upper_bound(path);
             it != prims_.end() && it->first.HasPrefix(path); ++it) {
            restamp(it->first, it->second);
        }
    }
}

void Scene::Stamp(const Path& path, PrimData& prim,
                  std::string_view attribute) {
    if (const auto it = prim.versions.find(attribute);
        it != prim.versions.end()) {
        it->second = version_;
    } else {
        prim.versions.emplace(std::string(attribute), version_);
    }
    if (IsXformAttribute(attribute)) {
        prim.ownTransform = version_;
    }
    if (TargetDirties().contains(attribute)) {
        staleDependents_.insert(path);
    }
    // Written alone, a column's row is the prim's own again.
    if (!prim.bound.empty()) {
        if (const auto bound = prim.bound.find(attribute);
            bound != prim.bound.end()) {
            const std::uint64_t column = bound->second.column;
            prim.bound.erase(bound);
            BreakColumn(column);
        }
    }
    Stamp(path, prim, DirtyOf(attribute), IsInherited(attribute));
}

void Scene::Refresh(const PrimData& prim,
                    std::string_view attribute) const {
    if (prim.bound.empty()) {
        return;
    }
    const auto bound = prim.bound.find(attribute);
    if (bound == prim.bound.end()) {
        return;
    }
    const auto found = columns_.find(bound->second.column);
    if (found == columns_.end() ||
        bound->second.seen >= found->second.version) {
        return;
    }
    const ColumnData& column = found->second;
    const std::int64_t row = bound->second.row;
    const auto assign = [&](auto& map, auto value) {
        if (const auto it = map.find(attribute); it != map.end()) {
            it->second = std::move(value);
        } else {
            map.emplace(std::string(attribute), std::move(value));
        }
    };
    assign(prim.streams, column.stream);
    if (column.offsets.empty()) {
        assign(prim.tensors, column.rows.Row(row));
        assign(prim.rows, std::pair(column.rows, row));
    } else {
        const auto first = static_cast<std::size_t>(row);
        assign(prim.tensors,
               column.rows.Slice(column.offsets[first],
                                 column.offsets[first + 1] -
                                     column.offsets[first]));
    }
    bound->second.seen = column.version;
}

void Scene::BreakColumn(std::uint64_t id) {
    const auto found = columns_.find(id);
    if (found == columns_.end()) {
        return;
    }
    const ColumnData column = std::move(found->second);
    columns_.erase(found);
    columnOf_.erase(column.key);
    const Dirty bit = DirtyOf(column.attribute);
    const bool descendants = IsInherited(column.attribute);
    for (const Path& path : *column.paths) {
        const auto it = prims_.find(path);
        if (it == prims_.end()) {
            continue;
        }
        PrimData& prim = it->second;
        const auto bound = prim.bound.find(column.attribute);
        if (bound == prim.bound.end() || bound->second.column != id) {
            continue;
        }
        // Its row as the column last held it, its own from now.
        if (bound->second.seen < column.version) {
            prim.tensors[column.attribute] =
                column.rows.Row(bound->second.row);
            prim.streams[column.attribute] = column.stream;
            prim.rows[column.attribute] =
                std::pair(column.rows, bound->second.row);
        }
        prim.bound.erase(bound);
        std::uint64_t& version = prim.versions[column.attribute];
        version = std::max(version, column.version);
        Stamp(path, prim, bit, descendants);
    }
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
        throw std::invalid_argument("Cannot define the prim at '" +
                                    path.String() + "'");
    }
    ++version_;
    const bool existed = Has(path);
    PrimData& prim = CreateIfMissing(path);
    if (prim.type != type) {
        if (existed) {
            // Its rows of columns gone, the columns end.
            std::vector<std::uint64_t> columns;
            for (const auto& [name, bound] : prim.bound) {
                columns.push_back(bound.column);
            }
            prim.bound.clear();
            for (const std::uint64_t column : columns) {
                BreakColumn(column);
            }
            prim.values.clear();
            prim.tensors.clear();
            prim.versions.clear();
            prim.samples.clear();
            animated_.erase(path);
            removed_[path] = version_;
            staleDependents_.insert(path);
            Stamp(path, prim, Dirty::Added);
            // Its transform and visibility are gone from under its
            // descendants too.
            Stamp(path, prim, Dirty::Transform);
            Stamp(path, prim, Dirty::Visibility);
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
    ERNIE_ZONE("Scene clone");
    if (!Has(source)) {
        throw std::invalid_argument("Cannot clone '" +
                                    source.String() +
                                    "', which does not exist");
    }
    if (destination.IsEmpty() || destination.IsRoot() ||
        Has(destination) || destination.HasPrefix(source)) {
        throw std::invalid_argument("Cannot clone '" +
                                    source.String() + "' to '" +
                                    destination.String() +
                                    "': the destination must be a "
                                    "new prim outside the source");
    }
    // Paths inside the source move with it.
    const auto remap = [&](const Path& path) {
        if (!path.HasPrefix(source)) {
            return path;
        }
        return Path(destination.String() +
                    path.String().substr(source.String().size()));
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
                Write(copy, attribute, *tensor,
                      WriteStream(path, attribute));
                continue;
            }
            const Value* value = Get(path, attribute);
            if (value == nullptr) {
                continue;
            }
            if (const auto* target = value->Get<Path>()) {
                Set(copy, attribute, remap(*target));
            } else if (const auto* targets =
                           value->Get<std::vector<Path>>()) {
                std::vector<Path> moved;
                for (const Path& each : *targets) {
                    moved.push_back(remap(each));
                }
                Set(copy, attribute, std::move(moved));
            } else if (const auto* text = value->Get<std::string>();
                       text != nullptr &&
                       attribute.ends_with(tokens::kConnectSuffix)) {
                // A shader connection, "prim path.output".
                const std::size_t dot = text->rfind('.');
                std::string moved = *text;
                if (dot != std::string::npos && dot > 0) {
                    try {
                        moved = remap(Path(text->substr(0, dot)))
                                    .String() +
                                text->substr(dot);
                    } catch (const std::invalid_argument&) {
                        // Not a prim path: copied as it is.
                    }
                }
                Set(copy, attribute, std::move(moved));
            } else {
                Set(copy, attribute, *value);
            }
        }
        // Its time samples, as they are (relationships are not
        // sampled).
        for (const auto& [attribute, samples] : Find(path).samples) {
            for (const TimeSample& sample : samples) {
                if (sample.isTensor) {
                    WriteAt(copy, attribute, sample.time,
                            sample.tensor, sample.stream);
                } else {
                    SetAt(copy, attribute, sample.time, sample.value);
                }
            }
        }
    }
    return GetPrim(destination);
}

bool Scene::Remove(const Path& path) {
    ERNIE_ZONE("Scene remove");
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
        if (dependencyTargets_.contains(last->first)) {
            staleDependents_.insert(last->first);
        }
        stamped_.erase({last->second.latest, last->first});
        animated_.erase(last->first);
        if (const auto typed = byType_.find(last->second.type);
            typed != byType_.end()) {
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

const PrimTypeDef* Scene::SchemaOf(const Path& path) const {
    return Find(path).schema;
}

std::vector<Path> Scene::Children(const Path& path) const {
    std::vector<Path> children;
    for (auto it = prims_.upper_bound(path);
         it != prims_.end() && it->first.HasPrefix(path); ++it) {
        if (it->first.Parent() == path) {
            children.push_back(it->first);
        }
    }
    return children;
}

std::vector<Path> Scene::Prims(std::string_view type) const {
    if (!type.empty()) {
        // By the index of each type's prims, in path order as the
        // walk below gives them.
        const auto typed = byType_.find(type);
        return typed == byType_.end()
                   ? std::vector<Path>{}
                   : std::vector<Path>(typed->second.begin(),
                                       typed->second.end());
    }
    std::vector<Path> paths;
    for (const auto& [path, prim] : prims_) {
        if (!path.IsRoot() && (type.empty() || prim.type == type)) {
            paths.push_back(path);
        }
    }
    return paths;
}

void Scene::Set(const Path& path, std::string_view attribute,
                Value value) {
    PrimData& prim = Find(path);
    Validate(prim.schema, path, attribute, &value);
    ++version_;
    if (const auto tensor = prim.tensors.find(attribute);
        tensor != prim.tensors.end()) {
        prim.tensors.erase(tensor);
    }
    if (const auto row = prim.rows.find(attribute);
        row != prim.rows.end()) {
        prim.rows.erase(row);
    }
    DropSamples(path, prim, attribute);
    if (const auto it = prim.values.find(attribute);
        it != prim.values.end()) {
        it->second = std::move(value);
    } else {
        prim.values.emplace(std::string(attribute), std::move(value));
    }
    Stamp(path, prim, attribute);
    // A new time code: every sampled attribute at it.
    if (path.IsRoot() && attribute == tokens::kTimeCode &&
        !animated_.empty()) {
        ERNIE_ZONE("Scene time code");
        const double time = TimeCode();
        for (const Path& animated : animated_) {
            PrimData& data = Find(animated);
            for (const auto& [name, samples] : data.samples) {
                Evaluate(animated, data, name, time);
            }
        }
    }
}

const Value* Scene::Get(const Path& path,
                        std::string_view attribute) const {
    const auto prim = prims_.find(path);
    if (prim == prims_.end()) {
        return nullptr;
    }
    const auto it = prim->second.values.find(attribute);
    return it != prim->second.values.end() ? &it->second : nullptr;
}

void Scene::Write(const Path& path, std::string_view attribute,
                  Tensor tensor, std::uintptr_t stream) {
    WriteTensor(path, Find(path), attribute, std::move(tensor),
                stream);
}

void Scene::WriteTensor(
    const Path& path, PrimData& prim, std::string_view attribute,
    Tensor tensor, std::uintptr_t stream,
    std::optional<std::pair<Tensor, std::int64_t>> row,
    bool checked) {
    if (!checked) {
        Validate(prim.schema, path, attribute, nullptr);
    }
    // Found before inserted, so rewriting an attribute every step
    // allocates no keys.
    if (const auto it = prim.streams.find(attribute);
        it != prim.streams.end()) {
        it->second = stream;
    } else {
        prim.streams.emplace(std::string(attribute), stream);
    }
    if (const auto it = prim.rows.find(attribute);
        it != prim.rows.end()) {
        if (row) {
            it->second = std::move(*row);
        } else {
            prim.rows.erase(it);
        }
    } else if (row) {
        prim.rows.emplace(std::string(attribute), std::move(*row));
    }
    ++version_;
    DropSamples(path, prim, attribute);
    if (const auto value = prim.values.find(attribute);
        value != prim.values.end()) {
        prim.values.erase(value);
    }
    if (const auto it = prim.tensors.find(attribute);
        it != prim.tensors.end()) {
        it->second = std::move(tensor);
    } else {
        prim.tensors.emplace(std::string(attribute),
                             std::move(tensor));
    }
    Stamp(path, prim, attribute);
}

Tensor Scene::Map(const Path& path, std::string_view attribute) {
    Read(path, attribute);  // Blended first, if pending.
    const PrimData& prim = Find(path);
    const auto it = prim.tensors.find(attribute);
    if (it == prim.tensors.end()) {
        throw std::invalid_argument(
            "Cannot map " + std::string(attribute) + " of '" +
            path.String() + "', which is not a written array");
    }
    return it->second;
}

void Scene::Unmap(const Path& path, std::string_view attribute,
                  std::uintptr_t stream) {
    PrimData& prim = Find(path);
    if (!prim.tensors.contains(attribute)) {
        throw std::invalid_argument(
            "Cannot unmap " + std::string(attribute) + " of '" +
            path.String() + "', which is not a written array");
    }
    ++version_;
    prim.streams[std::string(attribute)] = stream;
    Stamp(path, prim, attribute);
}

const Scene::ColumnData*
Scene::ColumnOf(const PrimSet& prims, std::string_view attribute) const {
    if (prims.id_ == 0 || prims.sceneId_ != Id() ||
        prims.removals_ != removals_ ||
        prims.prims_.size() != prims.Size()) {
        return nullptr;
    }
    const auto of =
        columnOf_.find(std::pair(prims.id_, std::string(attribute)));
    return of != columnOf_.end() ? &columns_.at(of->second) : nullptr;
}

Scene::ColumnData* Scene::ColumnOf(const PrimSet& prims,
                                   std::string_view attribute) {
    return const_cast<ColumnData*>(
        std::as_const(*this).ColumnOf(prims, attribute));
}

void Scene::Unmap(const PrimSet& prims, std::string_view attribute,
                  std::uintptr_t stream) {
    ERNIE_ZONE("Scene unmap rows");
    // A column written in place: its rows changed, at once.
    if (ColumnData* column = ColumnOf(prims, attribute)) {
        ++version_;
        column->stream = stream;
        column->version = version_;
        columnStamps_[StampIndex(DirtyOf(attribute))] = version_;
        return;
    }
    for (const Path& path : prims.Paths()) {
        if (!Find(path).tensors.contains(attribute)) {
            throw std::invalid_argument(
                "Cannot unmap " + std::string(attribute) + " of '" +
                path.String() + "', which is not a written array");
        }
    }
    for (const Path& path : prims.Paths()) {
        Unmap(path, attribute, stream);
    }
}

std::uintptr_t Scene::WriteStream(const Path& path,
                                  std::string_view attribute) const {
    if (!Has(path)) {
        return 1;
    }
    const PrimData& prim = Find(path);
    Refresh(prim, attribute);
    const auto it = prim.streams.find(attribute);
    return it != prim.streams.end() ? it->second : 1;
}

void Scene::Write(std::span<const Path> paths,
                  std::string_view attribute, const Tensor& rows,
                  std::uintptr_t stream) {
    Write(PrimSet(std::vector<Path>(paths.begin(), paths.end())),
          attribute, rows, stream);
}

void Scene::Write(const PrimSet& prims, std::string_view attribute,
                  const Tensor& rows, std::uintptr_t stream) {
    ERNIE_ZONE("Scene write rows");
    if (rows.Rank() == 0 ||
        rows.Shape()[0] != static_cast<std::int64_t>(prims.Size())) {
        throw std::invalid_argument(
            "Scene::Write needs one tensor row per path");
    }
    WriteColumn(prims, attribute, rows, {}, stream);
}

void Scene::Write(const PrimSet& prims, std::string_view attribute,
                  const Tensor& values,
                  std::span<const std::int64_t> counts,
                  std::uintptr_t stream) {
    ERNIE_ZONE("Scene write arrays");
    if (counts.size() != prims.Size()) {
        throw std::invalid_argument(
            "Scene::Write of arrays needs one count per path");
    }
    std::vector<std::int64_t> offsets{0};
    for (const std::int64_t count : counts) {
        if (count < 0) {
            throw std::invalid_argument(
                "Scene::Write of arrays needs counts of 0 or more");
        }
        offsets.push_back(offsets.back() + count);
    }
    if (values.Rank() == 0 || values.Shape()[0] != offsets.back()) {
        throw std::invalid_argument(
            "Scene::Write of arrays needs the counts to add up to the "
            "values' entries");
    }
    WriteColumn(prims, attribute, values, std::move(offsets), stream);
}

void Scene::WriteColumn(const PrimSet& prims, std::string_view attribute,
                        const Tensor& rows,
                        std::vector<std::int64_t> offsets,
                        std::uintptr_t stream) {
    // Laid out anew: arrays of other lengths than the column's, or
    // rows in place of arrays.
    if (const ColumnData* column = ColumnOf(prims, attribute);
        column != nullptr && column->offsets != offsets) {
        BreakColumn(columnOf_.at(column->key));
    }
    // The same prims' column again: replaced whole, its prims' rows
    // taken when read (`Refresh`), the change one `ColumnChange`.
    if (ColumnData* column = ColumnOf(prims, attribute)) {
        ++version_;
        column->provided = false;
        column->rows = rows;
        column->stream = stream;
        column->version = version_;
        columnStamps_[StampIndex(DirtyOf(attribute))] = version_;
        return;
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
    // The attribute is checked once per type among the prims
    // (`Validate`), not once per prim, and all before any is written,
    // so a rejected one writes nothing.
    std::vector<const PrimTypeDef*> checked;
    for (std::size_t i = 0; i < prims.Size(); ++i) {
        const PrimTypeDef* schema =
            static_cast<PrimData*>(prims.prims_[i])->schema;
        if (std::find(checked.begin(), checked.end(), schema) ==
            checked.end()) {
            Validate(schema, prims.Paths()[i], attribute, nullptr);
            checked.push_back(schema);
        }
    }
    for (std::size_t i = 0; i < prims.Size(); ++i) {
        const auto row = static_cast<std::int64_t>(i);
        // An array is a prim's own entries, not a row a consumer may
        // take with the others (`RowOf`).
        if (offsets.empty()) {
            WriteTensor(prims.Paths()[i],
                        *static_cast<PrimData*>(prims.prims_[i]),
                        attribute, rows.Row(row), stream,
                        std::pair(rows, row), true);
        } else {
            WriteTensor(prims.Paths()[i],
                        *static_cast<PrimData*>(prims.prims_[i]),
                        attribute,
                        rows.Slice(offsets[i], offsets[i + 1] - offsets[i]),
                        stream, std::nullopt, true);
        }
    }
    // From now one column, which the next write of the same prims
    // replaces whole (and `Unmap` of them marks changed whole).
    IdentifyPrimSet(prims);
    const std::uint64_t id = nextColumn_++;
    ColumnData column;
    column.paths = std::make_shared<const std::vector<Path>>(prims.Paths());
    column.sorted = prims.Paths();
    std::ranges::sort(column.sorted);
    column.attribute = std::string(attribute);
    column.rows = rows;
    column.stream = stream;
    column.version = version_;
    column.boundAt = version_;
    column.key = {prims.id_, std::string(attribute)};
    column.offsets = std::move(offsets);
    for (std::size_t i = 0; i < prims.Size(); ++i) {
        static_cast<PrimData*>(prims.prims_[i])
            ->bound.insert_or_assign(
                std::string(attribute),
                PrimData::Bound{id, static_cast<std::int64_t>(i),
                                version_});
    }
    columnOf_[column.key] = id;
    columns_.emplace(id, std::move(column));
}

namespace {

/// A value's numbers as a column's row holds them: vectors' components,
/// quaternions x, y, z, w (as arrays of them are), float arrays as
/// they are; none for matrices, text and paths.
std::optional<std::vector<float>> RowNumbers(const Value& value) {
    return std::visit(
        [](const auto& v) -> std::optional<std::vector<float>> {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, bool> ||
                          std::is_same_v<T, int> ||
                          std::is_same_v<T, float> ||
                          std::is_same_v<T, double>) {
                return std::vector<float>{static_cast<float>(v)};
            } else if constexpr (std::is_same_v<T, glm::dquat>) {
                return std::vector<float>{
                    static_cast<float>(v.x), static_cast<float>(v.y),
                    static_cast<float>(v.z), static_cast<float>(v.w)};
            } else if constexpr (std::is_same_v<T, glm::ivec2> ||
                                 std::is_same_v<T, glm::vec2> ||
                                 std::is_same_v<T, glm::vec3> ||
                                 std::is_same_v<T, glm::vec4> ||
                                 std::is_same_v<T, glm::dvec2> ||
                                 std::is_same_v<T, glm::dvec3> ||
                                 std::is_same_v<T, glm::dvec4>) {
                std::vector<float> numbers;
                for (glm::length_t i = 0; i < T::length(); ++i) {
                    numbers.push_back(static_cast<float>(v[i]));
                }
                return numbers;
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                return v;
            } else {
                return std::nullopt;
            }
        },
        value.Raw());
}

}  // namespace

Tensor Scene::GatherRows(const PrimSet& prims,
                         std::string_view attribute) const {
    std::vector<float> numbers;
    std::int64_t width = -1;
    for (const Path& path : prims.Paths()) {
        std::optional<std::vector<float>> row;
        if (const Tensor* tensor = Read(path, attribute)) {
            if (tensor->Location().kind != DeviceKind::Cpu) {
                throw std::invalid_argument(
                    std::string(attribute) + " of '" + path.String() +
                    "' is on a GPU: write the column's rows there");
            }
            const Tensor host = tensor->Contiguous();
            const auto count = static_cast<std::size_t>(host.Elements());
            if (host.Type() == DType::Float32) {
                row.emplace(host.As<float>(), host.As<float>() + count);
            } else if (host.Type() == DType::Float64) {
                row.emplace(host.As<double>(), host.As<double>() + count);
            }
        } else if (const Value* value = Get(path, attribute)) {
            row = RowNumbers(*value);
        }
        if (!row) {
            throw std::invalid_argument(
                "'" + path.String() + "' has no numbers as " +
                std::string(attribute) + " for a column's row");
        }
        const auto size = static_cast<std::int64_t>(row->size());
        if (width >= 0 && size != width) {
            throw std::invalid_argument(
                "The prims' " + std::string(attribute) +
                " differ in size, so they make no column");
        }
        width = size;
        numbers.insert(numbers.end(), row->begin(), row->end());
    }
    const auto rows = static_cast<std::int64_t>(prims.Size());
    return Tensor::Copy<float>(numbers, {rows, std::max<std::int64_t>(width, 0)});
}

Column Scene::GetColumn(const PrimSet& prims, std::string_view attribute) {
    // Its copies share its columns.
    IdentifyPrimSet(prims);
    return Column(*this, prims, std::string(attribute));
}

void Scene::SetColumnStorage(std::weak_ptr<ColumnStorage> storage) const {
    columnStorage_.storage = std::move(storage);
}

void Column::Write(const Tensor& rows, std::uintptr_t stream) {
    scene_->Write(prims_, attribute_, rows, stream);
}

void Column::WriteArrays(const Tensor& values,
                         std::span<const std::int64_t> counts,
                         std::uintptr_t stream) {
    scene_->Write(prims_, attribute_, values, counts, stream);
}

Tensor Column::Map(Device device, std::uintptr_t stream) {
    Scene& scene = *scene_;
    const auto held = [&]() -> const Tensor* {
        const Scene::ColumnData* data = scene.ColumnOf(prims_, attribute_);
        return data != nullptr && data->rows.Location() == device &&
                       data->rows.IsContiguous()
                   ? &data->rows
                   : nullptr;
    };
    // The consumer's own storage already: written where it reads.
    if (const Scene::ColumnData* data = scene.ColumnOf(prims_, attribute_);
        data != nullptr && data->provided && held() != nullptr) {
        return data->rows;
    }
    // Storage a consumer offers there, holding the rows as they are.
    if (const std::shared_ptr<ColumnStorage> storage =
            scene.columnStorage_.storage.lock()) {
        if (std::optional<ColumnStorage::Rows> offered =
                storage->Allocate(scene, prims_, attribute_, device, stream)) {
            if (const Tensor* now = held();
                now == nullptr || now->Data() != offered->rows.Data()) {
                scene.Write(prims_, attribute_, offered->rows, stream);
            }
            scene.ColumnOf(prims_, attribute_)->provided = offered->kept;
            return offered->rows;
        }
    }
    // The rows written last, if they are there.
    if (const Tensor* now = held()) {
        return *now;
    }
    // On the host, the prims' values gathered as rows.
    if (device.kind == DeviceKind::Cpu) {
        const Tensor rows = scene.GatherRows(prims_, attribute_);
        scene.Write(prims_, attribute_, rows, stream);
        return rows;
    }
    throw std::invalid_argument(
        "Nothing holds " + attribute_ +
        " of these prims on that device: write an array there first "
        "(Column::Write)");
}

void Column::Unmap(std::uintptr_t stream) {
    scene_->Unmap(prims_, attribute_, stream);
}

Tensor Column::Read() const {
    if (const Scene::ColumnData* data =
            std::as_const(*scene_).ColumnOf(prims_, attribute_)) {
        return data->rows;
    }
    return scene_->GatherRows(prims_, attribute_);
}

Tensor Column::Read(Device device) const {
    if (const Scene::ColumnData* data =
            std::as_const(*scene_).ColumnOf(prims_, attribute_);
        data != nullptr && data->rows.Location() == device) {
        return data->rows;
    }
    if (device.kind == DeviceKind::Cpu) {
        const Tensor rows = Read();
        if (rows.Location() == device) {
            return rows;
        }
    }
    throw std::invalid_argument(
        "The rows of " + attribute_ +
        " are not on that device: map or write them there first");
}

const std::pair<Tensor, std::int64_t>*
Scene::RowOf(const Path& path, std::string_view attribute) const {
    const auto prim = prims_.find(path);
    if (prim == prims_.end()) {
        return nullptr;
    }
    Refresh(prim->second, attribute);
    const auto it = prim->second.rows.find(attribute);
    return it != prim->second.rows.end() ? &it->second : nullptr;
}

std::uint64_t Scene::OwnTransformVersion(const Path& path) const {
    const auto prim = prims_.find(path);
    if (prim == prims_.end()) {
        return 0;
    }
    // Ops that are rows of columns change with them.
    std::uint64_t version = prim->second.ownTransform;
    for (const auto& [attribute, bound] : prim->second.bound) {
        if (const auto column = columns_.find(bound.column);
            column != columns_.end() && IsXformAttribute(attribute)) {
            version = std::max(version, column->second.version);
        }
    }
    return version;
}

const Tensor* Scene::Read(const Path& path,
                          std::string_view attribute) const {
    const auto prim = prims_.find(path);
    if (prim == prims_.end()) {
        return nullptr;
    }
    Refresh(prim->second, attribute);
    const auto it = prim->second.tensors.find(attribute);
    if (it == prim->second.tensors.end()) {
        return nullptr;
    }
    // A sampled tensor between samples, blended now it is read.
    if (!prim->second.pending.empty()) {
        if (const auto pending = prim->second.pending.find(attribute);
            pending != prim->second.pending.end()) {
            const std::vector<TimeSample>& samples =
                prim->second.samples.find(attribute)->second;
            if (std::optional<Tensor> blended =
                    Interpolate(samples[pending->second.lower].tensor,
                                samples[pending->second.upper].tensor,
                                pending->second.alpha)) {
                it->second = std::move(*blended);
            }
            prim->second.pending.erase(pending);
        }
    }
    return &it->second;
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
    DropSamples(path, prim, attribute);
    if (value != prim.values.end()) {
        prim.values.erase(value);
    }
    if (tensor != prim.tensors.end()) {
        prim.tensors.erase(tensor);
    }
    if (const auto row = prim.rows.find(attribute);
        row != prim.rows.end()) {
        prim.rows.erase(row);
    }
    Stamp(path, prim, attribute);
    return true;
}

std::uint64_t
Scene::AttributeVersion(const Path& path,
                        std::string_view attribute) const {
    const auto prim = prims_.find(path);
    if (prim == prims_.end()) {
        return 0;
    }
    const auto it = prim->second.versions.find(attribute);
    const std::uint64_t own =
        it != prim->second.versions.end() ? it->second : 0;
    // A column's row changes with the column.
    if (!prim->second.bound.empty()) {
        if (const auto bound = prim->second.bound.find(attribute);
            bound != prim->second.bound.end()) {
            if (const auto column = columns_.find(bound->second.column);
                column != columns_.end()) {
                return std::max(own, column->second.version);
            }
        }
    }
    return own;
}

glm::dmat4 Scene::LocalTransform(const Path& path) const {
    Find(path);
    return ComposeXformOps(*this, path);
}

glm::dmat4 Scene::WorldTransform(const Path& path) const {
    // Kept from the last call while nothing moved it: any transform
    // above it stamps it too (`Stamp`).
    const PrimData& prim = Find(path);
    if (prim.worldAt != 0 &&
        prim.worldAt >= prim.stamps[StampIndex(Dirty::Transform)] &&
        prim.worldAt >= columnStamps_[StampIndex(Dirty::Transform)]) {
        return prim.world;
    }
    glm::dmat4 world = ComposeXformOps(*this, path);
    if (const Path parent = path.Parent();
        !parent.IsEmpty() && !ResetsXformStack(*this, path)) {
        world = WorldTransform(parent) * world;
    }
    prim.world = world;
    prim.worldAt = version_;
    return world;
}

void Scene::SetLocalTransform(const Path& path,
                              const glm::dmat4& local) {
    for (auto& [attribute, value] :
         LocalTransformEdits(*this, path, local)) {
        Set(path, attribute, std::move(value));
    }
}

double Scene::TimeCode() const {
    const Value* time = Get(Path::Root(), tokens::kTimeCode);
    if (time == nullptr) {
        return 0.0;
    }
    if (const auto* d = time->Get<double>()) {
        return *d;
    }
    if (const auto* f = time->Get<float>()) {
        return *f;
    }
    if (const auto* i = time->Get<int>()) {
        return *i;
    }
    return 0.0;
}

void Scene::DropSamples(const Path& path, PrimData& prim,
                        std::string_view attribute) {
    const auto it = prim.samples.find(attribute);
    if (it == prim.samples.end()) {
        return;
    }
    prim.samples.erase(it);
    if (const auto pending = prim.pending.find(attribute);
        pending != prim.pending.end()) {
        prim.pending.erase(pending);
    }
    if (prim.samples.empty()) {
        animated_.erase(path);
    }
}

void Scene::AddSample(const Path& path, PrimData& prim,
                      std::string_view attribute, TimeSample sample) {
    ++version_;
    auto it = prim.samples.find(attribute);
    if (it == prim.samples.end()) {
        it = prim.samples
                 .emplace(std::string(attribute),
                          std::vector<TimeSample>())
                 .first;
    }
    std::vector<TimeSample>& samples = it->second;
    // One kind of sample per attribute.
    if (!samples.empty() &&
        samples.front().isTensor != sample.isTensor) {
        samples.clear();
    }
    const auto at = std::lower_bound(
        samples.begin(), samples.end(), sample.time,
        [](const TimeSample& s, double t) { return s.time < t; });
    if (at != samples.end() && at->time == sample.time) {
        *at = std::move(sample);
    } else {
        samples.insert(at, std::move(sample));
    }
    animated_.insert(path);
    Evaluate(path, prim, attribute, TimeCode());
}

void Scene::SetAt(const Path& path, std::string_view attribute,
                  double time, Value value) {
    PrimData& prim = Find(path);
    Validate(prim.schema, path, attribute, &value);
    TimeSample sample;
    sample.time = time;
    sample.value = std::move(value);
    AddSample(path, prim, attribute, std::move(sample));
}

void Scene::WriteAt(const Path& path, std::string_view attribute,
                    double time, Tensor tensor,
                    std::uintptr_t stream) {
    PrimData& prim = Find(path);
    Validate(prim.schema, path, attribute, nullptr);
    TimeSample sample;
    sample.time = time;
    sample.tensor = std::move(tensor);
    sample.stream = stream;
    sample.isTensor = true;
    AddSample(path, prim, attribute, std::move(sample));
}

void Scene::Evaluate(const Path& path, PrimData& prim,
                     std::string_view attribute, double time) {
    const std::vector<TimeSample>& samples =
        prim.samples.at(std::string(attribute));
    const SampleBracket bracket = BracketOf(samples, time);
    const TimeSample& lower = samples[bracket.lower];
    const TimeSample& upper = samples[bracket.upper];
    if (lower.isTensor) {
        // Between two samples it can blend: blended when first read
        // (`Read`), so consumers blending on the GPU (`BracketAt`)
        // cost the CPU nothing.
        const bool blends =
            bracket.alpha > 0.0 &&
            CanInterpolate(lower.tensor, upper.tensor);
        const auto pending = prim.pending.find(attribute);
        const auto it = prim.tensors.find(attribute);
        if (blends) {
            if (pending != prim.pending.end() &&
                pending->second.lower == bracket.lower &&
                pending->second.upper == bracket.upper &&
                pending->second.alpha == bracket.alpha) {
                return;
            }
            prim.pending[std::string(attribute)] = bracket;
        } else {
            if (pending == prim.pending.end() &&
                it != prim.tensors.end() &&
                it->second.Data() == lower.tensor.Data() &&
                it->second.Shape() == lower.tensor.Shape() &&
                it->second.Type() == lower.tensor.Type()) {
                return;  // The same sample as before.
            }
            if (pending != prim.pending.end()) {
                prim.pending.erase(pending);
            }
        }
        Tensor tensor = lower.tensor;  // Until blended, if it blends.
        const std::uintptr_t stream = blends ? 1 : lower.stream;
        if (const auto value = prim.values.find(attribute);
            value != prim.values.end()) {
            prim.values.erase(value);
        }
        if (const auto row = prim.rows.find(attribute);
            row != prim.rows.end()) {
            prim.rows.erase(row);
        }
        prim.streams[std::string(attribute)] = stream;
        if (it != prim.tensors.end()) {
            it->second = std::move(tensor);
        } else {
            prim.tensors.emplace(std::string(attribute),
                                 std::move(tensor));
        }
    } else {
        Value value =
            Interpolate(lower.value, upper.value, bracket.alpha);
        const auto it = prim.values.find(attribute);
        if (it != prim.values.end() && it->second == value) {
            return;
        }
        if (const auto tensor = prim.tensors.find(attribute);
            tensor != prim.tensors.end()) {
            prim.tensors.erase(tensor);
        }
        if (it != prim.values.end()) {
            it->second = std::move(value);
        } else {
            prim.values.emplace(std::string(attribute),
                                std::move(value));
        }
    }
    Stamp(path, prim, attribute);
}

std::span<const TimeSample>
Scene::SamplesOf(const Path& path, std::string_view attribute) const {
    const auto prim = prims_.find(path);
    if (prim == prims_.end()) {
        return {};
    }
    const auto it = prim->second.samples.find(attribute);
    return it != prim->second.samples.end()
               ? std::span<const TimeSample>(it->second)
               : std::span<const TimeSample>();
}

Value Scene::GetAt(const Path& path, std::string_view attribute,
                   double time) const {
    const std::span<const TimeSample> samples =
        SamplesOf(path, attribute);
    if (samples.empty()) {
        const Value* value = Get(path, attribute);
        return value != nullptr ? *value : Value();
    }
    if (samples.front().isTensor) {
        return {};
    }
    const SampleBracket bracket = BracketOf(samples, time);
    return Interpolate(samples[bracket.lower].value,
                       samples[bracket.upper].value, bracket.alpha);
}

bool Scene::ClearSamples(const Path& path,
                         std::string_view attribute) {
    PrimData& prim = Find(path);
    if (!prim.samples.contains(attribute)) {
        return false;
    }
    Read(path, attribute);  // Blended as it is now, if pending.
    ++version_;
    DropSamples(path, prim, attribute);
    return true;
}

std::optional<SampleBracket>
Scene::BracketAt(const Path& path, std::string_view attribute) const {
    const std::span<const TimeSample> samples =
        SamplesOf(path, attribute);
    if (samples.empty()) {
        return std::nullopt;
    }
    return BracketOf(samples, TimeCode());
}

std::vector<Path> Scene::AnimatedPrims() const {
    return {animated_.begin(), animated_.end()};
}

bool Scene::IsVisible(const Path& path) const {
    // Kept as `WorldTransform` keeps its matrix.
    const PrimData& prim = Find(path);
    if (prim.visibleAt != 0 &&
        prim.visibleAt >= prim.stamps[StampIndex(Dirty::Visibility)] &&
        prim.visibleAt >= columnStamps_[StampIndex(Dirty::Visibility)]) {
        return prim.visible;
    }
    const Value* visibility = Get(path, tokens::kVisibility);
    const std::string* token =
        visibility != nullptr ? visibility->Get<std::string>() : nullptr;
    // As a column's row, 0 hides it.
    double row = 1.0;
    const bool hidden = token != nullptr
                            ? *token == tokens::kInvisible
                            : GetRow(*this, path, tokens::kVisibility,
                                     {&row, 1}) &&
                                  row == 0.0;
    const Path parent = path.Parent();
    prim.visible = !hidden && (parent.IsEmpty() || IsVisible(parent));
    prim.visibleAt = version_;
    return prim.visible;
}

const std::map<Path, std::vector<std::pair<Path, Dirty>>>&
Scene::Dependents() const {
    if (staleDependents_.empty()) {
        return dependents_;
    }
    ERNIE_ZONE("Scene dependents");
    for (const Path& path : staleDependents_) {
        // Its entries as they were out...
        if (const auto old = dependencyTargets_.find(path);
            old != dependencyTargets_.end()) {
            for (const Path& target : old->second) {
                const auto it = dependents_.find(target);
                if (it == dependents_.end()) {
                    continue;
                }
                std::erase_if(it->second, [&path](const auto& entry) {
                    return entry.first == path;
                });
                if (it->second.empty()) {
                    dependents_.erase(it);
                }
            }
            dependencyTargets_.erase(old);
        }
        // ...and its relationships as they are in.
        const auto prim = prims_.find(path);
        if (prim == prims_.end()) {
            continue;
        }
        std::vector<Path> targets;
        for (const auto& [attribute, bit] : TargetDirties()) {
            const auto value = prim->second.values.find(attribute);
            if (value == prim->second.values.end()) {
                continue;
            }
            const auto depend = [&](const Path& target) {
                if (!target.IsEmpty()) {
                    dependents_[target].emplace_back(path, bit);
                    targets.push_back(target);
                }
            };
            if (const auto* target = value->second.Get<Path>()) {
                depend(*target);
            } else if (const auto* list =
                           value->second.Get<std::vector<Path>>()) {
                std::ranges::for_each(*list, depend);
            }
        }
        if (!targets.empty()) {
            dependencyTargets_.emplace(path, std::move(targets));
        }
    }
    staleDependents_.clear();
    return dependents_;
}

std::vector<Change> Scene::Changes(ChangeCursor& cursor) const {
    return Collect(cursor, nullptr);
}

std::vector<Change>
Scene::Changes(ChangeCursor& cursor,
               std::vector<ColumnChange>& columns) const {
    return Collect(cursor, &columns);
}

void Scene::AddColumn(std::map<Path, Dirty>& changed,
                      const std::vector<Path>& paths,
                      std::string_view attribute) const {
    const Dirty bit = DirtyOf(attribute);
    const bool descends = bit == Dirty::Transform ||
                          bit == Dirty::Visibility ||
                          IsInherited(attribute);
    // A prim added since reports `Added` alone.
    const auto add = [&changed, bit](const Path& path) {
        Dirty& bits = changed[path];
        if (!Any(bits, Dirty::Added)) {
            bits |= bit;
        }
    };
    for (const Path& path : paths) {
        const auto it = prims_.find(path);
        if (it == prims_.end()) {
            continue;
        }
        add(path);
        if (descends) {
            for (auto below = std::next(it);
                 below != prims_.end() && below->first.HasPrefix(path);
                 ++below) {
                add(below->first);
            }
        }
    }
}

void Scene::Forward(std::map<Path, Dirty>& changed) const {
    // Changes at or under a relationship's targets reach the prims
    // holding it, as a material's reach the prims bound to it, and on
    // through theirs (an animation, its skeleton, the skinned meshes).
    const auto& dependents = Dependents();
    for (bool more = !dependents.empty() && !changed.empty(); more;) {
        more = false;
        for (const auto& [target, sources] : dependents) {
            const auto at = changed.lower_bound(target);
            if (at == changed.end() || !at->first.HasPrefix(target)) {
                continue;
            }
            for (const auto& [source, bit] : sources) {
                if (prims_.contains(source)) {
                    Dirty& bits = changed[source];
                    more = more || !Any(bits, bit);
                    bits |= bit;
                }
            }
        }
    }
}

std::vector<Change>
Scene::ExpandColumns(std::span<const Change> changes,
                     std::span<const ColumnChange> columns) const {
    ERNIE_ZONE("Scene expand columns");
    std::map<Path, Dirty> changed;
    for (const Change& change : changes) {
        changed[change.path] |= change.bits;
    }
    for (const ColumnChange& column : columns) {
        AddColumn(changed, *column.paths, column.attribute);
    }
    Forward(changed);
    std::vector<Change> merged;
    merged.reserve(changed.size());
    for (auto& [path, bits] : changed) {
        merged.push_back({path, bits});
    }
    return merged;
}

std::vector<Change>
Scene::Collect(ChangeCursor& cursor,
               std::vector<ColumnChange>* columns) const {
    ERNIE_ZONE("Scene changes");
    std::map<Path, Dirty> changed;
    for (const auto& [path, when] : removed_) {
        if (when > cursor.version) {
            changed[path] = Dirty::Removed;
        }
    }
    // Only the prims stamped since the cursor, by `stamped_`.
    for (auto it = stamped_.lower_bound({cursor.version + 1, Path()});
         it != stamped_.end(); ++it) {
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
    // Columns written whole since the cursor (the write binding one
    // stamped its prims): whole for a consumer that takes them so,
    // unless some relationship's target is at, under or above their
    // prims, else as their prims' changes.
    const auto& dependents = Dependents();
    const auto reaches = [&dependents](const ColumnData& column) {
        for (const auto& [target, sources] : dependents) {
            // A row's prim at or under the target...
            const auto under = std::ranges::lower_bound(column.sorted, target);
            if (under != column.sorted.end() && under->HasPrefix(target)) {
                return true;
            }
            // ...or the target under one.
            for (Path at = target.Parent(); !at.IsEmpty() && !at.IsRoot();
                 at = at.Parent()) {
                if (std::ranges::binary_search(column.sorted, at)) {
                    return true;
                }
            }
        }
        return false;
    };
    for (const auto& [id, column] : columns_) {
        if (column.version <= cursor.version ||
            column.version == column.boundAt) {
            continue;
        }
        if (columns != nullptr && !reaches(column)) {
            columns->push_back({column.paths, column.attribute,
                                DirtyOf(column.attribute)});
        } else {
            AddColumn(changed, *column.paths, column.attribute);
        }
    }
    Forward(changed);
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
    return overlays_.emplace(std::string(name), Overlay())
        .first->second;
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

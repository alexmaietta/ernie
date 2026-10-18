// SPDX-License-Identifier: BSD-3-Clause
/// @file storage.h
/// @brief Columnar prim storage: prims in buckets by type and applied API schemas, each attribute
/// one array per bucket, with versions per bucket, column and 64-row chunk for change tracking.
#pragma once

#include "ernie/scene/path.h"
#include "ernie/util/bytes.h"
#include "ernie/util/glm.h"
#include "ernie/util/tensor.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ernie {

/// @brief A prim's handle: its slot, and the slot's generation when the prim was defined, so a
/// handle to a removed prim is detected rather than reaching its slot's next prim.
struct PrimId {
    std::uint32_t slot = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t generation = 0;

    bool operator==(const PrimId&) const = default;
};

using AttributeId = Token;
using BucketId = std::uint32_t;

/// @brief The element types columns hold. Bools are bytes, so a column is always a contiguous
/// array; tokens and path ids are the scene's interned names and paths.
enum class ColumnType : std::uint8_t {
    Bool,
    Int,
    Double,
    Vec2,
    Vec3,
    Vec4,
    Quat,
    Matrix,
    TokenValue,
    Relationship,
    Relationships,
    FloatArray,
    TokenArray,
    Tensor,
};

/// A relationship's single target, distinct from a token though both are 32-bit ids.
struct Target {
    PathId path = kNoPath;
    bool operator==(const Target&) const = default;
};

/// The element type of each column type, in `ColumnType`'s order.
using ColumnElements =
    std::tuple<std::uint8_t, std::int32_t, double, glm::dvec2, glm::dvec3, glm::dvec4, glm::dquat,
               glm::dmat4, Token, Target, std::vector<Target>, std::vector<float>,
               std::vector<Token>, Tensor>;
static_assert(std::tuple_size_v<ColumnElements> ==
              static_cast<std::size_t>(ColumnType::Tensor) + 1);

/// The column type of element type `T`.
template <typename T>
inline constexpr ColumnType kColumnType =
    []<typename... Ts>(std::type_identity<std::tuple<Ts...>>) {
        constexpr std::array matches{std::is_same_v<T, Ts>...};
        constexpr auto index =
            static_cast<std::size_t>(std::ranges::find(matches, true) - matches.begin());
        static_assert(index < matches.size(), "Not a column element type");
        return static_cast<ColumnType>(index);
    }(std::type_identity<ColumnElements>{});

/// @brief Calls `visit(std::type_identity<T>{})` with the element type `T` of a column type.
template <std::size_t I = 0, typename Visit>
decltype(auto) WithColumnType(ColumnType type, Visit&& visit) {
    if constexpr (I + 1 < std::tuple_size_v<ColumnElements>) {
        if (static_cast<std::size_t>(type) != I) {
            return WithColumnType<I + 1>(type, std::forward<Visit>(visit));
        }
    }
    return visit(std::type_identity<std::tuple_element_t<I, ColumnElements>>{});
}

/// Rows a version chunk covers: one word of the authored bits.
inline constexpr std::uint32_t kChunkRows = BitSet::kWordBits;

/// @brief A column's values, their element type erased.
class ColumnValues {
  public:
    virtual ~ColumnValues() = default;
    virtual ColumnType Type() const = 0;
    /// @return An empty column of the same type and fallback.
    virtual std::unique_ptr<ColumnValues> Clone() const = 0;
    /// Appends `count` rows holding the fallback.
    virtual void Grow(std::size_t count) = 0;
    /// Moves the last row into `row` and drops the last.
    virtual void SwapRemove(std::size_t row) = 0;
    /// Puts the fallback back in `row`.
    virtual void Reset(std::size_t row) = 0;
    /// Copies `source`'s value at `from` into `row`; `source` holds the same type.
    virtual void CopyRow(std::size_t row, const ColumnValues& source, std::size_t from) = 0;
};

template <typename T> class TypedColumn final : public ColumnValues {
  public:
    explicit TypedColumn(T fallback) : fallback_(std::move(fallback)) {}

    ColumnType Type() const override {
        return kColumnType<T>;
    }
    std::unique_ptr<ColumnValues> Clone() const override {
        return std::make_unique<TypedColumn<T>>(fallback_);
    }
    void Grow(std::size_t count) override {
        values_.resize(values_.size() + count, fallback_);
    }
    void SwapRemove(std::size_t row) override {
        if (row + 1 != values_.size()) {
            values_[row] = std::move(values_.back());
        }
        values_.pop_back();
    }
    void Reset(std::size_t row) override {
        values_[row] = fallback_;
    }
    void CopyRow(std::size_t row, const ColumnValues& source, std::size_t from) override {
        values_[row] = static_cast<const TypedColumn&>(source).values_[from];
    }

    std::vector<T>& Values() {
        return values_;
    }
    const std::vector<T>& Values() const {
        return values_;
    }

  private:
    std::vector<T> values_;
    T fallback_;
};

/// @brief One column of a bucket's layout: an attribute and how to make its values.
struct ColumnLayout {
    AttributeId attribute = 0;
    std::shared_ptr<const ColumnValues> prototype;

    /// @return A column of `T` whose unauthored rows hold `fallback`.
    template <typename T> static ColumnLayout Make(AttributeId attribute, T fallback = {}) {
        return {attribute, std::make_shared<TypedColumn<T>>(std::move(fallback))};
    }
};

/// @brief What a bucket groups: a prim type and its applied API schemas, kept sorted.
struct BucketKey {
    Token type = 0;
    std::vector<Token> apiSchemas;

    /// @return The key with its API schemas sorted and without repeats.
    BucketKey Normalized() const;
    bool operator==(const BucketKey&) const = default;
};

/// @brief A change to a bucket's rows.
struct StructuralChange {
    enum class Kind : std::uint8_t {
        /// `count` rows added from `row`.
        Added,
        /// The prim at `row` removed.
        Removed,
        /// The prim at `from` moved to `row` (the last row, into a removed one's place).
        Moved,
        /// The bucket's `count` rows, all new to a reader too far behind to replay its log.
        Reset,
    };
    std::uint64_t version = 0;
    Kind kind = Kind::Added;
    std::uint32_t row = 0;
    std::uint32_t count = 1;
    std::uint32_t from = 0;
};

/// @brief One attribute of every prim of a bucket, with its authored rows and versions.
struct Column {
    AttributeId attribute = 0;
    std::unique_ptr<ColumnValues> values;
    BitSet authored;
    /// The last version that changed any of chunk c's rows.
    std::vector<std::uint64_t> chunkVersions;
    std::uint64_t version = 0;

    /// Sizes the authored bits and chunk versions for `rows` rows.
    void Resize(std::uint32_t rows) {
        authored.Resize(rows);
        chunkVersions.resize(BitSet::WordCount(rows), 0);
    }
};

/// @brief Prims of one key, a row each, in structure-of-arrays columns.
struct Bucket {
    BucketKey key;
    /// The prim of each row.
    std::vector<PrimId> prims;
    std::vector<Column> columns;
    std::unordered_map<AttributeId, std::uint32_t> columnIndex;
    /// The last version that changed the bucket's rows or values.
    std::uint64_t version = 0;
    /// Row changes newer than `logStart`; older ones are dropped.
    std::deque<StructuralChange> log;
    std::uint64_t logStart = 0;

    std::uint32_t Rows() const {
        return static_cast<std::uint32_t>(prims.size());
    }
    /// @return The attribute's column, or null if the bucket has none.
    const Column* Find(AttributeId attribute) const {
        const auto found = columnIndex.find(attribute);
        return found == columnIndex.end() ? nullptr : &columns[found->second];
    }
    Column* Find(AttributeId attribute) {
        return const_cast<Column*>(std::as_const(*this).Find(attribute));
    }
};

/// @brief Rows [begin, end) of a column, changed.
struct RowRange {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    bool operator==(const RowRange&) const = default;
};

/// @brief What changed since a cursor (`PrimStore::ChangesSince`).
struct ChangeSet {
    /// The cursor to pass next time.
    std::uint64_t version = 0;
    struct Rows {
        BucketId bucket = 0;
        StructuralChange change;
    };
    struct Values {
        BucketId bucket = 0;
        AttributeId attribute = 0;
        std::vector<RowRange> rows;
    };
    /// Row changes in order.
    std::vector<Rows> rows;
    /// Changed values, by column.
    std::vector<Values> values;
};

/// @brief The scene's prims in columns.
/// One writer at a time; readers between writes. Every write stamps a new version; a reader's
/// cursor is the version it last saw, and `ChangesSince` returns what changed after it without
/// scanning unchanged buckets, columns or chunks.
class PrimStore {
  public:
    PrimStore() = default;
    PrimStore(const PrimStore&) = delete;
    PrimStore& operator=(const PrimStore&) = delete;
    PrimStore(PrimStore&&) = default;
    PrimStore& operator=(PrimStore&&) = default;

    PathTable& Paths() {
        return paths_;
    }
    const PathTable& Paths() const {
        return paths_;
    }
    TokenTable& Tokens() {
        return paths_.Names();
    }
    const TokenTable& Tokens() const {
        return paths_.Names();
    }

    /// @return The bucket of `key`, made with `layout`'s columns if new.
    BucketId AddBucket(const BucketKey& key, std::span<const ColumnLayout> layout);
    std::optional<BucketId> FindBucket(const BucketKey& key) const;
    const Bucket& GetBucket(BucketId bucket) const {
        return *buckets_[bucket];
    }
    std::size_t BucketCount() const {
        return buckets_.size();
    }

    /// @brief Defines a prim at `path` in `bucket`, its attributes at their fallbacks.
    /// @return The prim; the existing one if `path` already has one in `bucket`.
    /// @throws std::invalid_argument If `path` has a prim in another bucket.
    PrimId Define(PathId path, BucketId bucket);
    /// @brief Defines prims at `paths` in `bucket` as one write: one row change, one version.
    std::vector<PrimId> DefineMany(std::span<const PathId> paths, BucketId bucket);
    /// @brief Removes a prim; the bucket's last row moves into its place.
    void Remove(PrimId prim);
    /// @brief Moves a prim to another bucket, keeping its id and the authored values of the
    /// attributes both buckets have.
    void Move(PrimId prim, BucketId bucket);

    bool Alive(PrimId prim) const;
    std::optional<PrimId> Find(PathId path) const;
    PathId PrimPath(PrimId prim) const {
        return Slot(prim).path;
    }
    BucketId PrimBucket(PrimId prim) const {
        return Slot(prim).bucket;
    }
    std::uint32_t PrimRow(PrimId prim) const {
        return Slot(prim).row;
    }
    std::size_t PrimCount() const {
        return slots_.size() - free_.size();
    }

    /// @brief Authors an attribute of a prim, adding a column of `T` to its bucket if it has none.
    template <typename T> void Set(PrimId prim, AttributeId attribute, T value) {
        const SlotData& slot = Slot(prim);
        Column& column = FindOrAddColumn<T>(slot.bucket, attribute);
        Values<T>(column)[slot.row] = std::move(value);
        Stamp(slot.bucket, column, slot.row, slot.row + 1, Next());
    }
    /// @return The authored value, or null if unauthored or of another type.
    template <typename T> const T* Get(PrimId prim, AttributeId attribute) const {
        const SlotData& slot = Slot(prim);
        const Column* column = buckets_[slot.bucket]->Find(attribute);
        if (column == nullptr || !column->authored.Test(slot.row) ||
            column->values->Type() != kColumnType<T>) {
            return nullptr;
        }
        return &Values<T>(*column)[slot.row];
    }
    /// @brief Writes a value derived from others into a column the bucket has, leaving it
    /// unauthored and touching the row only when the value changes.
    /// @return Whether it changed; false too when the bucket has no such column.
    template <typename T> bool Derive(PrimId prim, AttributeId attribute, const T& value) {
        const SlotData& slot = Slot(prim);
        Column* column = buckets_[slot.bucket]->Find(attribute);
        if (column == nullptr || Values<T>(*column)[slot.row] == value) {
            return false;
        }
        Values<T>(*column)[slot.row] = value;
        Touch(slot.bucket, *column, slot.row, slot.row + 1, Next());
        return true;
    }
    /// @brief Unauthors an attribute, putting its fallback back.
    void Clear(PrimId prim, AttributeId attribute);

    /// @brief Writes a whole column at once: every row authored, one change of all rows.
    /// @return The values to fill, valid until the next structural change.
    template <typename T> std::span<T> Write(BucketId bucket, AttributeId attribute) {
        Column& column = FindOrAddColumn<T>(bucket, attribute);
        const std::uint32_t rows = buckets_[bucket]->Rows();
        Stamp(bucket, column, 0, rows, Next());
        return Values<T>(column);
    }
    /// @return A column's values, fallbacks where unauthored; empty if the bucket has none.
    template <typename T> std::span<const T> Read(BucketId bucket, AttributeId attribute) const {
        const Column* column = buckets_[bucket]->Find(attribute);
        return column == nullptr ? std::span<const T>() : std::span<const T>(Values<T>(*column));
    }

    std::uint64_t Version() const {
        return version_;
    }
    /// @return What changed after `cursor`; its `version` is the next cursor.
    ChangeSet ChangesSince(std::uint64_t cursor) const;

  private:
    struct SlotData {
        std::uint32_t generation = 0;
        BucketId bucket = 0;
        std::uint32_t row = 0;
        PathId path = kNoPath;
        bool alive = false;
    };
    struct KeyHash {
        std::size_t operator()(const BucketKey& key) const;
    };

    std::uint64_t Next() {
        return ++version_;
    }
    const SlotData& Slot(PrimId prim) const;
    SlotData& Slot(PrimId prim) {
        return const_cast<SlotData&>(std::as_const(*this).Slot(prim));
    }
    PrimId AddSlot(PathId path, BucketId bucket, std::uint32_t row);
    /// Appends `count` rows at their fallbacks to a bucket, logged as one change.
    std::uint32_t AddRows(BucketId bucket, std::uint32_t count);
    /// Drops a prim's row from its bucket, the last row moving into its place.
    void DropRow(const SlotData& slot);
    void Log(Bucket& bucket, StructuralChange change);
    /// Marks rows [begin, end) authored and changed at `version`.
    void Stamp(BucketId bucket, Column& column, std::uint32_t begin, std::uint32_t end,
               std::uint64_t version);
    /// Marks rows [begin, end) changed at `version`.
    void Touch(BucketId bucket, Column& column, std::uint32_t begin, std::uint32_t end,
               std::uint64_t version);
    Column& AddColumn(Bucket& bucket, AttributeId attribute, std::unique_ptr<ColumnValues> values);

    template <typename T> Column& FindOrAddColumn(BucketId bucket, AttributeId attribute) {
        Bucket& b = *buckets_[bucket];
        Column* found = b.Find(attribute);
        return found != nullptr ? *found
                                : AddColumn(b, attribute, std::make_unique<TypedColumn<T>>(T{}));
    }
    /// @throws std::invalid_argument if the column holds another type.
    template <typename T> static const std::vector<T>& Values(const Column& column) {
        if (column.values->Type() != kColumnType<T>) {
            throw std::invalid_argument("The column holds another type");
        }
        return static_cast<const TypedColumn<T>&>(*column.values).Values();
    }
    template <typename T> static std::vector<T>& Values(Column& column) {
        return const_cast<std::vector<T>&>(Values<T>(std::as_const(column)));
    }

    PathTable paths_;
    std::vector<std::unique_ptr<Bucket>> buckets_;
    std::unordered_map<BucketKey, BucketId, KeyHash> bucketIndex_;
    std::vector<SlotData> slots_;
    std::vector<std::uint32_t> free_;
    /// The prim slot of each path, or none.
    std::vector<std::uint32_t> pathSlots_;
    std::uint64_t version_ = 0;
};

}  // namespace ernie

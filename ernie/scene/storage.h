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

#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
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

template <typename T> struct ColumnTypeOf;
template <>
struct ColumnTypeOf<std::uint8_t> : std::integral_constant<ColumnType, ColumnType::Bool> {};
template <>
struct ColumnTypeOf<std::int32_t> : std::integral_constant<ColumnType, ColumnType::Int> {};
template <> struct ColumnTypeOf<double> : std::integral_constant<ColumnType, ColumnType::Double> {};
template <>
struct ColumnTypeOf<glm::dvec2> : std::integral_constant<ColumnType, ColumnType::Vec2> {};
template <>
struct ColumnTypeOf<glm::dvec3> : std::integral_constant<ColumnType, ColumnType::Vec3> {};
template <>
struct ColumnTypeOf<glm::dvec4> : std::integral_constant<ColumnType, ColumnType::Vec4> {};
template <>
struct ColumnTypeOf<glm::dquat> : std::integral_constant<ColumnType, ColumnType::Quat> {};
template <>
struct ColumnTypeOf<glm::dmat4> : std::integral_constant<ColumnType, ColumnType::Matrix> {};
template <>
struct ColumnTypeOf<Token> : std::integral_constant<ColumnType, ColumnType::TokenValue> {};
template <>
struct ColumnTypeOf<std::vector<Target>>
    : std::integral_constant<ColumnType, ColumnType::Relationships> {};
template <>
struct ColumnTypeOf<std::vector<float>>
    : std::integral_constant<ColumnType, ColumnType::FloatArray> {};
template <>
struct ColumnTypeOf<std::vector<Token>>
    : std::integral_constant<ColumnType, ColumnType::TokenArray> {};
template <> struct ColumnTypeOf<Tensor> : std::integral_constant<ColumnType, ColumnType::Tensor> {};

template <>
struct ColumnTypeOf<Target> : std::integral_constant<ColumnType, ColumnType::Relationship> {};

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
};

template <typename T> class TypedColumn final : public ColumnValues {
  public:
    explicit TypedColumn(T fallback) : fallback_(std::move(fallback)) {}

    ColumnType Type() const override {
        return ColumnTypeOf<T>::value;
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

    std::vector<T>& Values() {
        return values_;
    }
    const std::vector<T>& Values() const {
        return values_;
    }
    const T& Fallback() const {
        return fallback_;
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
    template <typename T> static ColumnLayout Of(AttributeId attribute, T fallback = {}) {
        return {attribute, std::make_shared<TypedColumn<T>>(std::move(fallback))};
    }
};

/// @brief What a bucket groups: a prim type and its applied API schemas, kept sorted.
struct BucketKey {
    Token type = 0;
    std::vector<Token> apiSchemas;

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
    /// Buckets whose changes are too old to replay: resync them whole.
    std::vector<BucketId> resets;
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

    /// @return The bucket of `key`, made with `layout`'s columns if new.
    BucketId AddBucket(const BucketKey& key, std::span<const ColumnLayout> layout);
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

    bool Alive(PrimId prim) const;
    std::optional<PrimId> Find(PathId path) const;
    PathId PathOf(PrimId prim) const {
        return Slot(prim).path;
    }
    BucketId BucketOf(PrimId prim) const {
        return Slot(prim).bucket;
    }
    std::uint32_t RowOf(PrimId prim) const {
        return Slot(prim).row;
    }
    std::size_t PrimCount() const {
        return slots_.size() - free_.size();
    }

    /// @brief Authors an attribute of a prim, adding a column of `T` to its bucket if it has none.
    template <typename T> void Set(PrimId prim, AttributeId attribute, T value) {
        const SlotData& slot = Slot(prim);
        Column& column = ColumnOf<T>(slot.bucket, attribute, true);
        Values<T>(column)[slot.row] = std::move(value);
        Stamp(slot.bucket, column, slot.row, slot.row + 1, Next());
    }
    /// @return The authored value, or null if unauthored.
    template <typename T> const T* Get(PrimId prim, AttributeId attribute) const {
        const SlotData& slot = Slot(prim);
        const Column* column = buckets_[slot.bucket]->Find(attribute);
        if (column == nullptr || !column->authored.Test(slot.row)) {
            return nullptr;
        }
        return &Values<T>(*column)[slot.row];
    }
    /// @brief Unauthors an attribute, putting its fallback back.
    void Clear(PrimId prim, AttributeId attribute);

    /// @brief Writes a whole column at once: every row authored, one change of all rows.
    /// @return The values to fill, valid until the next structural change.
    template <typename T> std::span<T> Write(BucketId bucket, AttributeId attribute) {
        Column& column = ColumnOf<T>(bucket, attribute, true);
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
    PrimId AddSlot(PathId path, BucketId bucket, std::uint32_t row);
    void Log(Bucket& bucket, StructuralChange change);
    /// Marks rows [begin, end) authored and changed at `version`.
    void Stamp(BucketId bucket, Column& column, std::uint32_t begin, std::uint32_t end,
               std::uint64_t version);
    Column& AddColumn(Bucket& bucket, AttributeId attribute, std::unique_ptr<ColumnValues> values);

    template <typename T> Column& ColumnOf(BucketId bucket, AttributeId attribute, bool add) {
        Bucket& b = *buckets_[bucket];
        const auto found = b.columnIndex.find(attribute);
        if (found != b.columnIndex.end()) {
            return b.columns[found->second];
        }
        if (!add) {
            throw std::out_of_range("The bucket has no such column");
        }
        return AddColumn(b, attribute, std::make_unique<TypedColumn<T>>(T{}));
    }
    template <typename T> static std::vector<T>& Values(Column& column) {
        return static_cast<TypedColumn<T>&>(Check<T>(column)).Values();
    }
    template <typename T> static const std::vector<T>& Values(const Column& column) {
        return static_cast<const TypedColumn<T>&>(Check<T>(column)).Values();
    }
    template <typename T> static const ColumnValues& Check(const Column& column) {
        if (column.values->Type() != ColumnTypeOf<T>::value) {
            throw std::invalid_argument("The column holds another type");
        }
        return *column.values;
    }
    template <typename T> static ColumnValues& Check(Column& column) {
        return const_cast<ColumnValues&>(Check<T>(static_cast<const Column&>(column)));
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

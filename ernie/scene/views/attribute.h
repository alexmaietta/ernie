// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <dlpack/dlpack.h>

#include "ernie/device/operation.h"
#include "ernie/device/ref.h"
#include "ernie/device/stream.h"
#include "ernie/device/tensor.h"
#include "ernie/export.h"
#include "ernie/scene/kernels/schema/schema.h"
#include "ernie/scene/primvar.h"
#include "ernie/scene/structure/tokens.h"
#include "ernie/scene/views/mapping.h"
#include "ernie/scene/views/view.h"
#include "ernie/util/error.h"
#include "ernie/util/span.h"

namespace ernie {

namespace device {
class Buffer;
}

class Store;
class Table;
struct Attachment;
struct ViewRows;

enum class AttachFrom : uint8_t { Tensor, Scene };

class ERNIE_API UntypedAttribute {
  public:
    UntypedAttribute(std::shared_ptr<ViewState> view, ColumnRef column);

    ColumnRef Reference() const {
        return m_column;
    }

    uint32_t Size() const;
    bool Dense() const;
    uint32_t CellBytes() const;

    void Fill(Span<const std::byte> cell);
    void Write(Span<const std::byte> cells);
    device::Operation Write(const DLTensor& source, Stream producer = Stream::None(),
                            const DLTensor* mask = nullptr);
    device::Operation Write(const device::Tensor& source, Stream producer = Stream::None(),
                            const device::Tensor& mask = {});

    device::Download ReadHostAsync() const;
    std::vector<std::byte> ReadHost() const;
    device::Tensor Read(Stream consumer = Stream::Host()) const;

    Mapping Map(MapAccess access, Stream consumer = Stream::None());
    device::Operation MarkChanged(Stream producer = Stream::None(), const DLTensor* mask = nullptr);

    device::Operation Attach(DLManagedTensor* tensor, Stream producer = Stream::None(),
                             AttachFrom from = AttachFrom::Tensor);
    device::Operation Attach(DLManagedTensorVersioned* tensor, Stream producer = Stream::None(),
                             AttachFrom from = AttachFrom::Tensor);
    device::Operation Detach(Stream consumer = Stream::None());
    device::Operation WaitForReads(Stream consumer = Stream::Host()) const;
    bool Attached() const;

    void SetArray(uint32_t index, Span<const std::byte> items);
    std::vector<std::byte> GetArray(uint32_t index) const;

    void SetTargets(uint32_t index, const std::vector<std::string>& paths);
    std::vector<std::string> GetTargets(uint32_t index) const;
    void SetTokens(uint32_t index, const std::vector<std::string>& texts);
    std::vector<std::string> GetTokens(uint32_t index) const;
    void SetPrimvars(uint32_t index, const std::vector<Primvar>& primvars);
    std::vector<Primvar> GetPrimvars(uint32_t index) const;

  protected:
    void CheckValues(size_t count) const;
    static ColumnRef Checked(ColumnRef column, bool held, bool arrays);

  private:
    friend class Mapping;
    friend class Poses;

    struct Mask {
        uint64_t address = 0;
        Ref<device::Buffer> buffer;
    };

    const ViewRows& Rows() const;
    Store& GetStore() const;
    Table& GetTable() const;
    void CheckWritable(bool arrays = false) const;
    void CheckNotAttached() const;
    void CheckArrayType(ColumnType type) const;
    void StageCells(Span<const std::byte> cells);
    device::Operation WriteTensor(const DLTensor& source, Stream producer, const DLTensor* mask);
    using Owner = std::function<std::shared_ptr<const DLTensor>()>;
    device::Operation AttachTensor(const DLTensor* source, Stream producer, AttachFrom from,
                                   const Owner& own);
    device::Operation AttachArrays(const Attachment& attachment, const DLTensor& source,
                                   Stream producer, AttachFrom from);
    device::Operation CopyArrays(const DLTensor& source, Stream producer, const DLTensor* mask);
    void ReleaseArrays();
    void MapArray(Mapping& mapping);
    void StampWorlds() const;
    Mask Upload(const DLTensor* mask, Stream producer) const;
    Mask Upload(const device::Tensor& mask) const;
    Ref<device::Buffer> GatherCells() const;
    device::Operation Scatter(uint64_t cells, Stream producer, const Mask& selected);
    device::Operation Stamp(Stream producer, const DLTensor* mask);

    std::shared_ptr<ViewState> m_view;
    ColumnRef m_column;
};

template <class T, class = void> struct HasItems : std::false_type {};

template <class T> struct HasItems<T, std::void_t<typename T::value_type>> : std::true_type {};

template <class T> struct Identity {
    using Type = T;
};

template <class T> using AttributeCell = std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>;

template <class T, class = void> struct AttributeValue;

template <class T> struct AttributeValue<T, std::enable_if_t<std::is_enum_v<T>>> {
    static constexpr ColumnType kType = ColumnType::Enum;
};

template <> struct AttributeValue<Token> {
    static constexpr ColumnType kType = ColumnType::Token;
};

template <> struct AttributeValue<bool> {
    static constexpr ColumnType kType = ColumnType::Bool;
};

template <> struct AttributeValue<int32_t> {
    static constexpr ColumnType kType = ColumnType::Int;
};

template <> struct AttributeValue<uint32_t> {
    static constexpr ColumnType kType = ColumnType::Uint;
};

template <> struct AttributeValue<uint64_t> {
    static constexpr ColumnType kType = ColumnType::Uint64;
};

template <> struct AttributeValue<float> {
    static constexpr ColumnType kType = ColumnType::Float;
};

template <> struct AttributeValue<Float2> {
    static constexpr ColumnType kType = ColumnType::Float2;
};

template <> struct AttributeValue<Float3> {
    static constexpr ColumnType kType = ColumnType::Float3;
};

template <> struct AttributeValue<Float4> {
    static constexpr ColumnType kType = ColumnType::Float4;
};

template <> struct AttributeValue<Uint2> {
    static constexpr ColumnType kType = ColumnType::Uint2;
};

template <> struct AttributeValue<Double3> {
    static constexpr ColumnType kType = ColumnType::Double3;
};

template <> struct AttributeValue<Quatd> {
    static constexpr ColumnType kType = ColumnType::Quatd;
};

template <> struct AttributeValue<WorldTransform> {
    static constexpr ColumnType kType = ColumnType::WorldTransform;
};

template <class T> constexpr bool HeldAs(ColumnType type) {
    if constexpr (std::is_same_v<T, Token>) {
        return type == ColumnType::Token || type == ColumnType::Asset;
    } else {
        return type == AttributeValue<T>::kType;
    }
}

template <class T> class Attribute : public UntypedAttribute {
  public:
    Attribute(std::shared_ptr<ViewState> view, ColumnRef column)
        : UntypedAttribute(std::move(view), Check(column)) {}

    void Set(const T& value) {
        const Cell cell = Cell(value);
        Fill(AsBytes(Span<const Cell>(&cell, 1)));
    }

    void Set(Span<const T> values) {
        if (values.size() != Size()) {
            CheckValues(values.size());
        }
        if constexpr (std::is_same_v<T, bool>) {
            const std::vector<uint32_t> cells(values.begin(), values.end());
            Write(AsBytes(Span<const uint32_t>(cells)));
        } else {
            Write(AsBytes(values));
        }
    }

    void Set(const std::vector<T>& values) {
        if constexpr (std::is_same_v<T, bool>) {
            const std::vector<uint32_t> cells(values.begin(), values.end());
            CheckValues(cells.size());
            Write(AsBytes(Span<const uint32_t>(cells)));
        } else {
            Set(Span<const T>(values));
        }
    }

    std::vector<T> Get() const {
        const std::vector<Cell> cells = Items<Cell>(ReadHost());
        if constexpr (std::is_same_v<T, Cell>) {
            return cells;
        } else {
            return std::vector<T>(cells.begin(), cells.end());
        }
    }

  private:
    using Cell = AttributeCell<T>;

    static ColumnRef Check(ColumnRef column) {
        return Checked(column, HeldAs<T>(column.column->type), false);
    }
};

template <class T> class ArrayAttribute : public UntypedAttribute {
  public:
    ArrayAttribute(std::shared_ptr<ViewState> view, ColumnRef column)
        : UntypedAttribute(std::move(view), Check(column)) {}

    void Set(uint32_t index, Span<const T> items) {
        SetArray(index, AsBytes(items));
    }

    void Set(uint32_t index, const std::vector<T>& items) {
        Set(index, Span<const T>(items));
    }

    std::vector<T> Get(uint32_t index) const {
        return Items<T>(GetArray(index));
    }

  private:
    static ColumnRef Check(ColumnRef column) {
        return Checked(column,
                       IsArray(column.column->type) && ItemBytes(column.column->type) == sizeof(T),
                       true);
    }
};

template <class T> ArrayAttribute<T> View::GetArray(ColumnRef column) const {
    if (!m_state) {
        throw Error("An empty view has no attributes");
    }
    return ArrayAttribute<T>(m_state, column);
}

template <class T> Attribute<T> View::Get(ColumnRef column) const {
    if (!m_state) {
        throw Error("An empty view has no attributes");
    }
    return Attribute<T>(m_state, column);
}

template <class T> auto View::Get(Field<T> field) const {
    const ColumnRef column = field.Column();
    if constexpr (std::is_same_v<T, std::string>) {
        return Get<Token>(column);
    } else if constexpr (std::is_same_v<T, std::vector<std::string>> ||
                         std::is_same_v<T, std::vector<Primvar>>) {
        return Get(column);
    } else if constexpr (HasItems<T>::value) {
        return GetArray<typename T::value_type>(column);
    } else {
        return Get<T>(column);
    }
}

}

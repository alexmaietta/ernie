// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <dlpack/dlpack.h>

#if !defined(DLPACK_MAJOR_VERSION) || DLPACK_MAJOR_VERSION < 1
#error "Ernie needs DLPack 1.0 or later, but an older dlpack.h was included first"
#endif

#include "ernie/device/operation.h"
#include "ernie/device/ref.h"
#include "ernie/export.h"
#include "ernie/util/span.h"

namespace ernie::device {

class Buffer;
class Device;

enum class DType : uint8_t { UInt8, UInt32, Float32, Float64, Int32, UInt64, Int64 };

constexpr uint32_t ElementBytes(DType dtype) {
    return dtype == DType::UInt8                                                        ? 1
           : dtype == DType::Float64 || dtype == DType::UInt64 || dtype == DType::Int64 ? 8
                                                                                        : 4;
}

template <class T> constexpr DType ElementType() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return DType::UInt8;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return DType::UInt32;
    } else if constexpr (std::is_same_v<T, float>) {
        return DType::Float32;
    } else if constexpr (std::is_same_v<T, double>) {
        return DType::Float64;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DType::Int32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return DType::Int64;
    } else {
        static_assert(std::is_same_v<T, uint64_t>, "Tensors hold no elements of this type");
        return DType::UInt64;
    }
}

class ERNIE_API Tensor {
  public:
    Tensor();
    ~Tensor();
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(Ref<Buffer> buffer, uint64_t offset, DType dtype, std::vector<int64_t> shape,
           std::vector<int64_t> strides = {});

    bool Empty() const {
        return m_buffer == nullptr;
    }

    Buffer& GetBuffer() const {
        return *m_buffer;
    }

    Device& GetDevice() const;

    uint64_t Offset() const {
        return m_offset;
    }

    uint64_t Address() const;

    DType Type() const {
        return m_dtype;
    }

    Span<const int64_t> Shape() const {
        return m_shape;
    }

    Span<const int64_t> Strides() const {
        return m_strides;
    }

    int64_t Elements() const;
    Tensor Select(int64_t index) const;

    std::vector<std::byte> ToHostBytes() const;
    Download ToHostAsync() const;
    DLTensor DLPack() const;
    DLManagedTensorVersioned* Export() const;

    template <class T> std::vector<T> ToHost() const {
        if (sizeof(T) != ElementBytes(m_dtype)) {
            throw std::invalid_argument("ToHost of a type unlike the tensor's elements");
        }
        std::vector<T> values(static_cast<size_t>(Elements()));
        if (!values.empty()) {
            ReadInto(values.data());
        }
        return values;
    }

  private:
    void ReadInto(void* out) const;

    Ref<Buffer> m_buffer;
    uint64_t m_offset = 0;
    DType m_dtype = DType::Float32;
    std::vector<int64_t> m_shape;
    std::vector<int64_t> m_strides;
};

}

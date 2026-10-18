// SPDX-License-Identifier: BSD-3-Clause
/// @file tensor.h
/// @brief Strided tensors, Ernie's end of DLPack.
#pragma once

#include "ernie/export.h"

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ernie {

/// @brief Element types Ernie exchanges.
enum class DType : std::uint8_t { UInt8, UInt16, UInt32, Int32, Int64, Float16, Float32, Float64 };

/// @brief The size of an element type.
/// @param type The type.
/// @return Bytes per element.
ERNIE_API std::size_t ByteSize(DType type);

/// @brief Where a tensor's memory lives: host, CUDA or ROCm (HIP).
enum class DeviceKind : std::uint8_t { Cpu, Cuda, Rocm };

/// @param kind The memory kind.
/// @return Whether it is a GPU compute device's memory.
constexpr bool IsGpu(DeviceKind kind) {
    return kind != DeviceKind::Cpu;
}

/// @brief A memory location: a kind and a device index.
struct Device {
    /// The kind of memory.
    DeviceKind kind = DeviceKind::Cpu;
    /// The device index within its kind.
    int id = 0;

    bool operator==(const Device&) const = default;
};

/// @brief A strided n-dimensional view of typed memory, with an
/// optional owner keeping it alive.
/// Shapes and strides are in elements; empty strides mean row-major
/// contiguous. Copies share memory and owner. DLPack tensors are
/// adopted and exported without copying.
///
/// References:
/// - DLPack, https://dmlc.github.io/dlpack/latest/
class ERNIE_API Tensor {
  public:
    Tensor() = default;

    /// @brief Views memory.
    /// @param data    The first element.
    /// @param type    The element type.
    /// @param shape   The extent of each dimension.
    /// @param strides Strides in elements; empty for contiguous.
    /// @param device  Where the memory lives.
    /// @param owner   Kept alive as long as any copy of the view.
    /// @throws std::invalid_argument If a dimension is negative or
    ///         strides do not match the rank.
    Tensor(void* data, DType type, std::vector<std::int64_t> shape,
           std::vector<std::int64_t> strides = {}, Device device = {},
           std::shared_ptr<const void> owner = {});

    /// @brief Allocates a zero-filled contiguous CPU tensor.
    /// @param type  The element type.
    /// @param shape The extent of each dimension.
    /// @return The owning tensor.
    static Tensor Zeros(DType type, std::vector<std::int64_t> shape);

    /// @brief Copies values into a new contiguous CPU tensor.
    /// @param values The elements, row-major.
    /// @param shape  The extent of each dimension.
    /// @return The owning tensor.
    /// @throws std::invalid_argument If the shape's element count
    ///         differs from `values.size()`.
    template <typename T>
    static Tensor Copy(std::span<const T> values, std::vector<std::int64_t> shape);

    /// @brief Adopts a DLPack tensor without copying.
    /// Its deleter runs when the last referencing `Tensor` goes.
    /// @param managed The producer's tensor.
    /// @return The tensor.
    /// @throws std::invalid_argument For an unsupported type, lanes
    ///         or device; the deleter still runs.
    static Tensor FromDlpack(DLManagedTensorVersioned* managed);
    /// @brief Adopts an unversioned DLPack tensor, as above.
    /// @param managed The producer's tensor.
    /// @return The tensor.
    static Tensor FromDlpack(DLManagedTensor* managed);

    /// @brief Exports to DLPack without copying.
    /// @return A tensor keeping the memory alive until the consumer
    ///         calls its deleter.
    DLManagedTensorVersioned* ToDlpack() const;

    /// @return Whether it views no memory.
    bool Empty() const {
        return data_ == nullptr;
    }

    /// @return The first element.
    void* Data() const {
        return data_;
    }

    /// @return The element type.
    DType Type() const {
        return type_;
    }

    /// @return Where the memory lives.
    Device Location() const {
        return device_;
    }

    /// @return The extent of each dimension.
    const std::vector<std::int64_t>& Shape() const {
        return shape_;
    }

    /// @return Strides in elements, filled in when contiguous.
    const std::vector<std::int64_t>& Strides() const {
        return strides_;
    }

    /// @return The number of dimensions.
    std::size_t Rank() const {
        return shape_.size();
    }

    /// @return The number of elements.
    std::int64_t Elements() const;

    /// @brief Views one entry along the first dimension.
    /// @param index The entry.
    /// @return A view sharing memory and owner.
    /// @throws std::out_of_range If rank 0 or `index` is out of
    ///         range.
    Tensor Row(std::int64_t index) const;

    /// @brief Views a range along the first dimension.
    /// @param first The first entry.
    /// @param count How many entries.
    /// @return A view sharing memory and owner.
    /// @throws std::out_of_range If rank 0 or the range is out of
    ///         range.
    Tensor Slice(std::int64_t first, std::int64_t count) const;

    /// @brief Views contiguous memory with another shape, on any
    /// device.
    /// @param shape The new shape, of as many elements.
    /// @return A view sharing memory and owner.
    /// @throws std::invalid_argument If not contiguous or the element
    ///         counts differ.
    Tensor Reshaped(std::vector<std::int64_t> shape) const;

    /// @return Whether the elements are row-major contiguous.
    bool IsContiguous() const;

    /// @brief A contiguous CPU tensor of the same elements.
    /// @return This tensor if already contiguous, else an owning
    ///         copy.
    /// @throws std::invalid_argument If it is not on the CPU.
    Tensor Contiguous() const;

    /// @brief Typed access to the first element.
    /// @return The pointer.
    /// @throws std::invalid_argument If `T` is not the element type.
    template <typename T> T* As() const;

  private:
    void* data_ = nullptr;
    DType type_ = DType::Float32;
    std::vector<std::int64_t> shape_;
    std::vector<std::int64_t> strides_;
    Device device_;
    std::shared_ptr<const void> owner_;
};

/// @brief The `DType` of C++ element type `T`.
template <typename T>
inline constexpr DType kDType = [] {
    if constexpr (std::is_same_v<T, std::uint8_t>) {
        return DType::UInt8;
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
        return DType::UInt16;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        return DType::UInt32;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return DType::Int32;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return DType::Int64;
    } else if constexpr (std::is_same_v<T, float>) {
        return DType::Float32;
    } else if constexpr (std::is_same_v<T, double>) {
        return DType::Float64;
    } else {
        static_assert(sizeof(T) == 0, "Not a tensor element type");
    }
}();

template <typename T>
Tensor Tensor::Copy(std::span<const T> values, std::vector<std::int64_t> shape) {
    Tensor tensor = Zeros(kDType<T>, std::move(shape));
    if (static_cast<std::size_t>(tensor.Elements()) != values.size()) {
        throw std::invalid_argument("Tensor::Copy shape does not match the number of values");
    }
    std::copy(values.begin(), values.end(), tensor.As<T>());
    return tensor;
}

template <typename T> T* Tensor::As() const {
    if (kDType<T> != type_) {
        throw std::invalid_argument("Tensor::As with the wrong element type");
    }
    return static_cast<T*>(data_);
}

/// @brief Copies a CPU tensor of any shape out as floats.
/// @param tensor A float32 or float64 tensor on the CPU.
/// @param what   What it holds, for errors.
/// @return Its elements in row-major order.
/// @throws std::invalid_argument If it is elsewhere or not float.
ERNIE_API std::vector<float> CopyFloats(const Tensor& tensor, std::string_view what);

/// @brief Copies a CPU array of vectors out as floats.
/// @param tensor     A flat or (n, `components`) float tensor.
/// @param components The vectors' size.
/// @param what       What it holds, for errors.
/// @return Its elements in row-major order.
/// @throws std::invalid_argument If its shape or type is wrong.
ERNIE_API std::vector<float> CopyFloats(const Tensor& tensor, std::size_t components,
                                        std::string_view what);

/// @brief Copies a CPU tensor of any shape out as integers.
/// @param tensor An int32, int64 or uint32 tensor on the CPU.
/// @param what   What it holds, for errors.
/// @return Its elements in row-major order.
/// @throws std::invalid_argument If it is elsewhere or not integer.
ERNIE_API std::vector<std::int64_t> CopyInts(const Tensor& tensor, std::string_view what);

}  // namespace ernie

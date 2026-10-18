// SPDX-License-Identifier: BSD-3-Clause
/// @file dlpack.h
/// @brief Tensors to and from Python's DLPack protocol (`__dlpack__`), shared rather than copied.
#pragma once

#include "ernie/util/tensor.h"

#include <nanobind/nanobind.h>

namespace ernie::python {

/// @brief Wraps a DLPack tensor in the capsule `__dlpack__` returns.
/// @param managed   The tensor; the capsule owns it until a consumer takes it.
/// @param versioned Whether the consumer asked for a versioned capsule.
/// @return The capsule.
nanobind::object Capsule(DLManagedTensorVersioned* managed, bool versioned);

/// @brief Takes a tensor from any DLPack producer, such as a NumPy array or a PyTorch tensor.
/// @param source The producer.
/// @param stream The stream its data must be ready on, as `__dlpack__` takes it; None for the
///               host or the default stream.
/// @return The tensor, sharing the producer's memory.
Tensor TakeTensor(const nanobind::object& source, const nanobind::object& stream);

}  // namespace ernie::python

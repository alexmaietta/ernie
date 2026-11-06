// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "bindings.h"

namespace ernie::python {

struct ScenePrim {
    Ref<Scene> scene;
    Prim prim;
};

template <class T> std::vector<std::byte> Bytes(const std::vector<T>& values) {
    const auto bytes = std::as_bytes(std::span(values));
    return {bytes.begin(), bytes.end()};
}

std::vector<std::byte> NumpyBytes(nanobind::handle value, const std::string& dtype);
std::vector<std::string> Strings(nanobind::handle value);
uint32_t EnumIndex(nanobind::handle value, std::span<const std::string_view> tokens);

void Assign(bool& out, nanobind::handle value);
void Assign(int32_t& out, nanobind::handle value);
void Assign(uint32_t& out, nanobind::handle value);
void Assign(float& out, nanobind::handle value);
void Assign(Float2& out, nanobind::handle value);
void Assign(Uint2& out, nanobind::handle value);
void Assign(Float3& out, nanobind::handle value);
void Assign(Float4& out, nanobind::handle value);
void Assign(Double3& out, nanobind::handle value);
void Assign(Quatd& out, nanobind::handle value);
void Assign(std::string& out, nanobind::handle value);
void Assign(std::vector<std::string>& out, nanobind::handle value);
void Assign(std::vector<int32_t>& out, nanobind::handle value);
void Assign(std::vector<int64_t>& out, nanobind::handle value);
void Assign(std::vector<Float3>& out, nanobind::handle value);
void Assign(std::vector<float>& out, nanobind::handle value);
void Assign(std::vector<Float2>& out, nanobind::handle value);
void Assign(std::vector<Float4>& out, nanobind::handle value);
void Assign(std::vector<Primvar>& out, nanobind::handle value);

template <class E>
    requires std::is_enum_v<E>
void Assign(E& out, nanobind::handle value, std::span<const std::string_view> tokens) {
    out = E(EnumIndex(value, tokens));
}

Prim DefineSchema(Scene& scene, std::string_view path, SchemaId schema,
                  const nanobind::dict& values, const nanobind::dict& apis);
void ApplySchema(Scene& scene, Span<const Prim> prims, SchemaId schema, std::string_view instance,
                 const nanobind::dict& values);

}

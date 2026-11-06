// SPDX-License-Identifier: BSD-3-Clause
#include "values.h"

#include <cstring>

#include <nanobind/stl/map.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace ernie::python {

namespace {

template <class T> void AssignCell(T& out, nb::handle value, const char* dtype) {
    const std::vector<std::byte> bytes = NumpyBytes(value, dtype);
    if (bytes.size() != sizeof(T)) {
        throw nb::value_error(("Expected " + std::to_string(sizeof(T) / 4) + " values, not " +
                               std::to_string(bytes.size() / 4))
                                  .c_str());
    }
    std::memcpy(&out, bytes.data(), sizeof(T));
}

}

std::vector<std::byte> NumpyBytes(nb::handle value, const std::string& dtype) {
    const nb::object array =
        nb::module_::import_("numpy").attr("ascontiguousarray")(value, "dtype"_a = dtype);
    const nb::bytes bytes = nb::cast<nb::bytes>(array.attr("tobytes")());
    const auto* first = reinterpret_cast<const std::byte*>(bytes.c_str());
    return {first, first + bytes.size()};
}

std::vector<std::string> Strings(nb::handle value) {
    if (nb::isinstance<nb::str>(value)) {
        return {nb::cast<std::string>(value)};
    }
    std::vector<std::string> strings;
    for (nb::handle item : value) {
        strings.push_back(nb::isinstance<ScenePrim>(item) ? nb::cast<ScenePrim&>(item).prim.Path()
                                                          : nb::cast<std::string>(item));
    }
    return strings;
}

uint32_t EnumIndex(nb::handle value, std::span<const std::string_view> tokens) {
    if (!nb::isinstance<nb::str>(value)) {
        return nb::cast<uint32_t>(value);
    }
    const std::string text = nb::cast<std::string>(value);
    for (uint32_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index] == text) {
            return index;
        }
    }
    std::string allowed;
    for (const std::string_view token : tokens) {
        allowed += (allowed.empty() ? "" : ", ") + std::string(token);
    }
    throw nb::value_error((text + " isn't one of " + allowed).c_str());
}

void Assign(bool& out, nb::handle value) {
    out = nb::cast<bool>(value);
}

void Assign(int32_t& out, nb::handle value) {
    out = nb::cast<int32_t>(value);
}

void Assign(uint32_t& out, nb::handle value) {
    out = nb::cast<uint32_t>(value);
}

void Assign(float& out, nb::handle value) {
    out = nb::cast<float>(value);
}

void Assign(Float2& out, nb::handle value) {
    AssignCell(out, value, "float32");
}

void Assign(Uint2& out, nb::handle value) {
    AssignCell(out, value, "uint32");
}

void Assign(Float3& out, nb::handle value) {
    AssignCell(out, value, "float32");
}

void Assign(Float4& out, nb::handle value) {
    AssignCell(out, value, "float32");
}

void Assign(Double3& out, nb::handle value) {
    AssignCell(out, value, "float64");
}

void Assign(Quatd& out, nb::handle value) {
    AssignCell(out, value, "float64");
}

void Assign(std::string& out, nb::handle value) {
    out = nb::cast<std::string>(value);
}

void Assign(std::vector<std::string>& out, nb::handle value) {
    out = Strings(value);
}

void Assign(std::vector<int32_t>& out, nb::handle value) {
    out = Items<int32_t>(NumpyBytes(value, "int32"));
}

void Assign(std::vector<int64_t>& out, nb::handle value) {
    out = Items<int64_t>(NumpyBytes(value, "int64"));
}

void Assign(std::vector<float>& out, nb::handle value) {
    out = Items<float>(NumpyBytes(value, "float32"));
}

void Assign(std::vector<Float3>& out, nb::handle value) {
    out = Items<Float3>(NumpyBytes(value, "float32"));
}

void Assign(std::vector<Float2>& out, nb::handle value) {
    out = Items<Float2>(NumpyBytes(value, "float32"));
}

void Assign(std::vector<Float4>& out, nb::handle value) {
    out = Items<Float4>(NumpyBytes(value, "float32"));
}

void Assign(std::vector<Primvar>& out, nb::handle value) {
    out = nb::cast<std::vector<Primvar>>(value);
}

}

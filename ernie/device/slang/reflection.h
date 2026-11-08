// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/device/adapter.h"
#include "ernie/device/slang/compiler.h"

namespace ernie::device {

enum class FieldType { Bool, Int, Float, Float2, Float3, Float4 };

struct ReflectedField {
    std::string name;
    FieldType type = FieldType::Float;
    std::vector<std::byte> fallback;
    bool color = false;
    float minimum = 0;
    float maximum = 0;
    std::string displayName;
};

struct ReflectedNode {
    std::string name;
    std::vector<ReflectedField> fields;
};

ERNIE_API std::vector<ReflectedNode>
ReflectMaterialNodes(Backend backend, const KernelLibrary* library, const SourceFiles& sources,
                     std::string_view module, std::string_view label);

}

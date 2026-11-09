// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ernie/export.h"
#include "ernie/stage/schema/schema.h"

namespace ernie {

struct RuntimeColumnDesc {
    std::string field;
    ColumnType type = ColumnType::Float;
    ColumnRole role = ColumnRole::None;
    RuntimeValue fallback = 0.0f;
    std::string displayName;
    std::string group;
    std::string doc;
    float minimum = 0;
    float maximum = 0;
};

struct RuntimeSchemaDesc {
    std::string name;
    SchemaId base = SchemaId::None;
    std::string doc;
    std::vector<RuntimeColumnDesc> columns;
    std::vector<std::pair<std::string, RuntimeValue>> inherited;
};

ERNIE_API SchemaId RegisterSchema(const RuntimeSchemaDesc& desc, bool replace = false);

ERNIE_API SchemaId FindSchemaByName(std::string_view name);

}

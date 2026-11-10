// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/stage/schema/types.h"

#ifndef __SLANG__

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "ernie/export.h"
#include "ernie/util/span.h"

namespace ernie {

enum class SchemaId : uint32_t;

enum class SchemaKind : uint8_t { Abstract, Concrete, Api };

enum class ColumnType : uint8_t {
    Bool,
    Int,
    Uint,
    Uint64,
    Float,
    Float2,
    Uint2,
    Float3,
    Double3,
    Quatd,
    WorldTransform,
    Enum,
    Token,
    IntArray,
    Float3Array,
    PrimArray,
    Asset,
    Float4,
    TokenArray,
    Float4Array,
    Float2Array,
    FloatArray,
    Int64Array,
};

enum class ColumnRole : uint8_t { None, Color, Point, Normal, Vector };

enum class Unit : uint8_t {
    None,
    Meters,
    Millimeters,
    Degrees,
    Radians,
    Kelvin,
    Seconds,
    Nits,
    Lux,
    Watts,
    Hertz,
};

enum class ColumnFlags : uint8_t {
    None = 0,
    Internal = 1 << 0,
};

constexpr ColumnFlags operator|(ColumnFlags a, ColumnFlags b) {
    return ColumnFlags(uint8_t(a) | uint8_t(b));
}

constexpr bool Has(ColumnFlags flags, ColumnFlags bits) {
    return (uint8_t(flags) & uint8_t(bits)) != 0;
}

constexpr bool IsArray(ColumnType type) {
    return type == ColumnType::IntArray || type == ColumnType::Float3Array ||
           type == ColumnType::Float4Array || type == ColumnType::Float2Array ||
           type == ColumnType::FloatArray || type == ColumnType::PrimArray ||
           type == ColumnType::TokenArray || type == ColumnType::Int64Array;
}

constexpr uint32_t ElementBytes(ColumnType type) {
    switch (type) {
    case ColumnType::Uint64:
        return sizeof(uint64_t);
    case ColumnType::Float2:
        return sizeof(Float2);
    case ColumnType::Uint2:
        return sizeof(Uint2);
    case ColumnType::Float3:
        return sizeof(Float3);
    case ColumnType::Float4:
        return sizeof(Float4);
    case ColumnType::Double3:
        return sizeof(Double3);
    case ColumnType::Quatd:
        return sizeof(Quatd);
    case ColumnType::WorldTransform:
        return sizeof(WorldTransform);
    case ColumnType::IntArray:
    case ColumnType::Float3Array:
    case ColumnType::Float4Array:
    case ColumnType::Float2Array:
    case ColumnType::FloatArray:
    case ColumnType::PrimArray:
    case ColumnType::TokenArray:
    case ColumnType::Int64Array:
        return sizeof(ArrayRef);
    default:
        return 4;
    }
}

constexpr uint32_t ItemBytes(ColumnType type) {
    switch (type) {
    case ColumnType::IntArray:
    case ColumnType::FloatArray:
    case ColumnType::PrimArray:
    case ColumnType::TokenArray:
        return 4;
    case ColumnType::Float3Array:
        return sizeof(Float3);
    case ColumnType::Float4Array:
        return sizeof(Float4);
    case ColumnType::Float2Array:
        return sizeof(Float2);
    case ColumnType::Int64Array:
        return 8;
    default:
        return 0;
    }
}

struct ColumnInfo {
    std::string_view field;
    std::string_view name;
    ColumnType type;
    ColumnRole role;
    ColumnFlags flags;
    uint32_t offset;
    SchemaId api;
    std::string_view doc;
    Unit unit;
    Span<const std::string_view> tokens{};
    std::string_view group{};
    std::string_view displayName{};
    float minimum = 0;
    float maximum = 0;
    std::string_view enabledBy{};
};

struct SchemaInfo {
    std::string_view name;
    SchemaId base;
    SchemaKind kind;
    uint32_t firstApi;
    uint32_t apiCount;
    uint32_t firstColumn;
    uint32_t columnCount;
    uint32_t dataBytes;
    uint32_t firstSlot;
    uint32_t slotCount;
    uint32_t api;
    uint64_t apis;
    uint32_t firstTarget;
    uint32_t targetCount;
    std::string_view doc;
};

inline constexpr uint32_t kNoApi = 0xFFFFFFFF;

struct BuiltinInstanceInfo {
    SchemaId schema;
    SchemaId api;
    std::string_view instance;
};

struct ColumnRef;

template <class T> struct Field {
    SchemaId owner;
    uint32_t index;

    constexpr ColumnRef Column() const;
};

using RuntimeValue = std::variant<bool, int32_t, float, Float2, Float3, Float4, std::string>;

ERNIE_API uint32_t SchemaCount();
ERNIE_API const SchemaInfo& RuntimeInfo(SchemaId schema);
ERNIE_API Span<const ColumnInfo> RuntimeColumns(SchemaId schema);
ERNIE_API Span<const SchemaId> RuntimeSlots(SchemaId schema);
ERNIE_API const RuntimeValue& RuntimeFallback(SchemaId schema, uint32_t column);

struct RuntimeInheritance {
    SchemaId owner;
    uint32_t index;
    RuntimeValue value;
};

ERNIE_API Span<const RuntimeInheritance> RuntimeInherited(SchemaId schema);

}

#endif

#include "ernie/scene/schema/registry.h"

#ifndef __SLANG__

namespace ernie {

constexpr bool IsRuntime(SchemaId schema) {
    return uint32_t(schema) >= uint32_t(SchemaId::Count);
}

constexpr const SchemaInfo& Info(SchemaId schema) {
    return IsRuntime(schema) ? RuntimeInfo(schema) : kSchemaInfo[uint32_t(schema)];
}

constexpr Span<const SchemaId> Apis(SchemaId schema) {
    return Span<const SchemaId>(kSchemaApis).subspan(Info(schema).firstApi, Info(schema).apiCount);
}

constexpr Span<const ColumnInfo> Columns(SchemaId schema) {
    if (IsRuntime(schema)) {
        return RuntimeColumns(schema);
    }
    return Span<const ColumnInfo>(kColumns).subspan(Info(schema).firstColumn,
                                                    Info(schema).columnCount);
}

inline constexpr uint32_t kNoSlot = 0xFFFFFFFF;

constexpr Span<const SchemaId> Slots(SchemaId schema) {
    if (IsRuntime(schema)) {
        return RuntimeSlots(schema);
    }
    return Span<const SchemaId>(kSchemaSlots)
        .subspan(Info(schema).firstSlot, Info(schema).slotCount);
}

constexpr uint32_t Slot(SchemaId schema, SchemaId table) {
    const Span<const SchemaId> slots = Slots(schema);
    for (uint32_t slot = 0; slot < slots.size(); ++slot) {
        if (slots[slot] == table) {
            return slot;
        }
    }
    return kNoSlot;
}

constexpr bool IsA(SchemaId schema, SchemaId base) {
    for (; schema != SchemaId::None; schema = Info(schema).base) {
        if (schema == base) {
            return true;
        }
    }
    return false;
}

constexpr bool IsApi(SchemaId schema) {
    return Info(schema).api != kNoApi;
}

constexpr uint64_t ApiBit(SchemaId api) {
    return uint64_t(1) << Info(api).api;
}

constexpr bool IsMultipleApi(SchemaId schema) {
    return IsApi(schema) && (kMultipleApis & ApiBit(schema)) != 0;
}

constexpr bool BuiltinInstance(SchemaId schema, SchemaId api, std::string_view instance) {
    for (const BuiltinInstanceInfo& entry : kBuiltinInstances) {
        if (entry.schema == schema && entry.api == api && entry.instance == instance) {
            return true;
        }
    }
    return false;
}

constexpr bool HasApi(uint64_t apis, SchemaId api) {
    return IsApi(api) && (apis & ApiBit(api)) != 0;
}

constexpr bool Includes(SchemaId schema, SchemaId api) {
    return HasApi(Info(schema).apis, api);
}

constexpr Span<const SchemaId> ApiTargets(SchemaId api) {
    return Span<const SchemaId>(kApiTargets).subspan(Info(api).firstTarget, Info(api).targetCount);
}

constexpr bool AppliesTo(SchemaId api, SchemaId schema) {
    if (!IsApi(api) || Info(schema).kind == SchemaKind::Api) {
        return false;
    }
    const Span<const SchemaId> targets = ApiTargets(api);
    if (targets.empty()) {
        return true;
    }
    for (const SchemaId target : targets) {
        if (IsA(schema, target)) {
            return true;
        }
    }
    return false;
}

constexpr uint32_t ApiSlot(SchemaId schema, uint64_t apis, SchemaId api) {
    if (!HasApi(apis, api)) {
        return kNoSlot;
    }
    uint32_t below = 0;
    for (uint64_t bits = apis & (ApiBit(api) - 1); bits != 0; bits &= bits - 1) {
        ++below;
    }
    return Info(schema).slotCount + below;
}

constexpr uint32_t PrimSlot(SchemaId schema, uint64_t apis, SchemaId table) {
    return IsApi(table) ? ApiSlot(schema, apis, table) : Slot(schema, table);
}

inline SchemaId ShaderSchema(std::string_view id) {
    for (uint32_t index = SchemaCount(); index-- > 0;) {
        const SchemaId schema = SchemaId(index);
        if (Info(schema).name == id && Info(schema).kind == SchemaKind::Concrete &&
            IsA(schema, SchemaId::Shader)) {
            return schema;
        }
    }
    return SchemaId::None;
}

inline std::vector<SchemaId> PrimTables(SchemaId schema, uint64_t apis) {
    const Span<const SchemaId> typed = Slots(schema);
    std::vector<SchemaId> tables(typed.begin(), typed.end());
    for (const SchemaId api : kApiSchemas) {
        if ((apis & ApiBit(api)) != 0) {
            tables.push_back(api);
        }
    }
    return tables;
}

struct ColumnRef {
    SchemaId schema = SchemaId::None;
    const ColumnInfo* column = nullptr;
    uint32_t index = 0;

    explicit constexpr operator bool() const {
        return column != nullptr;
    }
};

template <class T> constexpr ColumnRef Field<T>::Column() const {
    return {owner, &Columns(owner)[index], index};
}

constexpr ColumnRef FindOwnColumn(SchemaId schema, std::string_view field) {
    const Span<const ColumnInfo> columns = Columns(schema);
    for (uint32_t index = 0; index < columns.size(); ++index) {
        if (columns[index].field == field) {
            return {schema, &columns[index], index};
        }
    }
    return {};
}

constexpr ColumnRef FindColumn(SchemaId schema, std::string_view field) {
    for (; schema != SchemaId::None; schema = Info(schema).base) {
        if (const ColumnRef own = FindOwnColumn(schema, field)) {
            return own;
        }
        for (const SchemaId api : Apis(schema)) {
            if (const ColumnRef builtin = FindOwnColumn(api, field)) {
                return builtin;
            }
        }
    }
    return {};
}

constexpr ColumnRef FindColumn(SchemaId schema, uint64_t apis, std::string_view field) {
    if (const ColumnRef found = FindColumn(schema, field)) {
        return found;
    }
    for (const SchemaId api : kApiSchemas) {
        if ((apis & ApiBit(api)) != 0) {
            if (const ColumnRef applied = FindOwnColumn(api, field)) {
                return applied;
            }
        }
    }
    return {};
}

template <class W> void WriteRuntimeFallbacks(SchemaId schema, W& w) {
    if (!IsRuntime(schema)) {
        return;
    }
    WriteFallbacks(Info(schema).base, w);
    if (IsA(schema, SchemaId::Shader)) {
        w.Set(SchemaId::Shader, FindOwnColumn(SchemaId::Shader, "id").index,
              std::string(Info(schema).name));
    }
    for (const RuntimeInheritance& inherited : RuntimeInherited(schema)) {
        std::visit([&](const auto& value) { w.Set(inherited.owner, inherited.index, value); },
                   inherited.value);
    }
    for (uint32_t column = 0; column < Info(schema).columnCount; ++column) {
        std::visit([&](const auto& value) { w.Set(schema, column, value); },
                   RuntimeFallback(schema, column));
    }
}

}

#endif

// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/schema/runtime.h"

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>

#include "ernie/util/error.h"
#include "ernie/util/log.h"

namespace ernie {

namespace {

constexpr uint32_t kMaxRuntimeSchemas = 4096;

struct RuntimeSchema {
    RuntimeSchemaDesc desc;
    std::deque<std::string> strings;
    SchemaInfo info{};
    std::vector<ColumnInfo> columns;
    std::vector<SchemaId> slots;
    std::vector<RuntimeValue> fallbacks;
    std::vector<RuntimeInheritance> inherited;

    std::string_view Keep(std::string text) {
        return strings.emplace_back(std::move(text));
    }
};

struct Registry {
    std::mutex mutex;
    std::array<std::unique_ptr<RuntimeSchema>, kMaxRuntimeSchemas> schemas;
    std::atomic<uint32_t> count{0};
};

Registry& RuntimeRegistry() {
    static Registry registry;
    return registry;
}

const RuntimeSchema& Lookup(SchemaId schema) {
    const uint32_t index = uint32_t(schema) - uint32_t(SchemaId::Count);
    Registry& registry = RuntimeRegistry();
    if (index >= registry.count.load(std::memory_order_acquire)) {
        throw Error("No schema {}", uint32_t(schema));
    }
    return *registry.schemas[index];
}

bool Accepts(ColumnType type, const RuntimeValue& value) {
    switch (type) {
    case ColumnType::Bool:
        return std::holds_alternative<bool>(value);
    case ColumnType::Int:
    case ColumnType::Enum:
        return std::holds_alternative<int32_t>(value);
    case ColumnType::Float:
        return std::holds_alternative<float>(value);
    case ColumnType::Float2:
        return std::holds_alternative<Float2>(value);
    case ColumnType::Float3:
        return std::holds_alternative<Float3>(value);
    case ColumnType::Float4:
        return std::holds_alternative<Float4>(value);
    case ColumnType::Token:
    case ColumnType::Asset:
        return std::holds_alternative<std::string>(value);
    default:
        return false;
    }
}

bool Same(const RuntimeSchemaDesc& a, const RuntimeSchemaDesc& b) {
    if (a.name != b.name || a.base != b.base || a.doc != b.doc ||
        a.columns.size() != b.columns.size() || a.inherited != b.inherited) {
        return false;
    }
    for (size_t i = 0; i < a.columns.size(); ++i) {
        const RuntimeColumnDesc& x = a.columns[i];
        const RuntimeColumnDesc& y = b.columns[i];
        if (x.field != y.field || x.type != y.type || x.role != y.role ||
            x.fallback != y.fallback || x.displayName != y.displayName || x.group != y.group ||
            x.doc != y.doc || x.minimum != y.minimum || x.maximum != y.maximum) {
            return false;
        }
    }
    return true;
}

void Check(const RuntimeSchemaDesc& desc) {
    if (desc.name.empty()) {
        throw Error("A runtime schema needs a name");
    }
    if (desc.base == SchemaId::None || IsApi(desc.base) ||
        Info(desc.base).kind == SchemaKind::Api) {
        throw Error("Runtime schema {} needs a typed base schema", desc.name);
    }
    for (size_t i = 0; i < desc.columns.size(); ++i) {
        const RuntimeColumnDesc& column = desc.columns[i];
        if (column.field.empty()) {
            throw Error("{}'s input {} has no name", desc.name, i);
        }
        if (FindColumn(desc.base, column.field)) {
            throw Error("{}'s input {} hides {}'s attribute of that name", desc.name, column.field,
                        Info(desc.base).name);
        }
        for (size_t j = 0; j < i; ++j) {
            if (desc.columns[j].field == column.field) {
                throw Error("{} has two inputs called {}", desc.name, column.field);
            }
        }
        if (!Accepts(column.type, column.fallback)) {
            throw Error("{}'s input {} has a fallback of the wrong type", desc.name, column.field);
        }
    }
    for (const auto& [field, value] : desc.inherited) {
        const ColumnRef column = FindColumn(desc.base, field);
        if (!column) {
            throw Error("{} sets {}, which {} doesn't have", desc.name, field,
                        Info(desc.base).name);
        }
        if (!Accepts(column.column->type, value)) {
            throw Error("{} sets {} to a value of the wrong type", desc.name, field);
        }
    }
}

std::unique_ptr<RuntimeSchema> Build(const RuntimeSchemaDesc& desc, SchemaId id) {
    auto schema = std::make_unique<RuntimeSchema>();
    schema->desc = desc;
    for (const SchemaId slot : Slots(desc.base)) {
        schema->slots.push_back(slot);
    }
    if (!desc.columns.empty()) {
        schema->slots.push_back(id);
    }
    for (const RuntimeColumnDesc& column : desc.columns) {
        schema->columns.push_back({
            .field = schema->Keep(column.field),
            .name = schema->Keep("inputs:" + column.field),
            .type = column.type,
            .role = column.role,
            .flags = ColumnFlags::Tracked,
            .offset = 0,
            .stampOffset = 0,
            .api = SchemaId::None,
            .doc = schema->Keep(column.doc),
            .unit = Unit::None,
            .tokens = {},
            .group = schema->Keep(column.group),
            .displayName =
                schema->Keep(column.displayName.empty() ? column.field : column.displayName),
            .minimum = column.minimum,
            .maximum = column.maximum,
        });
        schema->fallbacks.push_back(column.fallback);
    }
    for (const auto& [field, value] : desc.inherited) {
        const ColumnRef column = FindColumn(desc.base, field);
        schema->inherited.push_back({column.schema, column.index, value});
    }
    schema->info = {
        .name = schema->Keep(desc.name),
        .base = desc.base,
        .kind = SchemaKind::Concrete,
        .firstApi = 0,
        .apiCount = 0,
        .firstColumn = 0,
        .columnCount = uint32_t(desc.columns.size()),
        .dataBytes = 0,
        .firstSlot = 0,
        .slotCount = uint32_t(schema->slots.size()),
        .api = kNoApi,
        .apis = Info(desc.base).apis,
        .firstTarget = 0,
        .targetCount = 0,
        .doc = schema->Keep(desc.doc),
    };
    return schema;
}

}

uint32_t SchemaCount() {
    return uint32_t(SchemaId::Count) + RuntimeRegistry().count.load(std::memory_order_acquire);
}

const SchemaInfo& RuntimeInfo(SchemaId schema) {
    return Lookup(schema).info;
}

Span<const ColumnInfo> RuntimeColumns(SchemaId schema) {
    const RuntimeSchema& found = Lookup(schema);
    return Span<const ColumnInfo>(found.columns.data(), found.columns.size());
}

Span<const SchemaId> RuntimeSlots(SchemaId schema) {
    const RuntimeSchema& found = Lookup(schema);
    return Span<const SchemaId>(found.slots.data(), found.slots.size());
}

const RuntimeValue& RuntimeFallback(SchemaId schema, uint32_t column) {
    return Lookup(schema).fallbacks.at(column);
}

Span<const RuntimeInheritance> RuntimeInherited(SchemaId schema) {
    const RuntimeSchema& found = Lookup(schema);
    return Span<const RuntimeInheritance>(found.inherited.data(), found.inherited.size());
}

SchemaId FindSchemaByName(std::string_view name) {
    const uint32_t count = SchemaCount();
    for (uint32_t index = 0; index < count; ++index) {
        if (Info(SchemaId(index)).name == name) {
            return SchemaId(index);
        }
    }
    return SchemaId::None;
}

SchemaId RegisterSchema(const RuntimeSchemaDesc& desc) {
    Check(desc);
    Registry& registry = RuntimeRegistry();
    const std::lock_guard lock(registry.mutex);
    if (const SchemaId existing = FindSchemaByName(desc.name); existing != SchemaId::None) {
        if (IsRuntime(existing) && Same(Lookup(existing).desc, desc)) {
            return existing;
        }
        throw Error("A schema called {} already exists", desc.name);
    }
    const uint32_t index = registry.count.load(std::memory_order_relaxed);
    if (index >= kMaxRuntimeSchemas) {
        throw Error("Registering {} would pass the limit of {} runtime schemas", desc.name,
                    kMaxRuntimeSchemas);
    }
    const SchemaId id = SchemaId(uint32_t(SchemaId::Count) + index);
    registry.schemas[index] = Build(desc, id);
    registry.count.store(index + 1, std::memory_order_release);
    Log(LogLevel::Debug, "registered schema {}, a {} with {} inputs", desc.name,
        Info(desc.base).name, desc.columns.size());
    return id;
}

}

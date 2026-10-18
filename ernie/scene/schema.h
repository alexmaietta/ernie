// SPDX-License-Identifier: BSD-3-Clause
/// @file schema.h
/// @brief The schema registry: prim types, API schemas and their attributes, declared once.
#pragma once

#include "ernie/geometry/bounds.h"
#include "ernie/scene/storage.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/value.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// What changed about a prim, as bit flags; each attribute declares which it raises.
enum class Dirty : std::uint32_t {
    None = 0,
    Added = 1u << 0,
    Removed = 1u << 1,
    Transform = 1u << 2,
    Topology = 1u << 3,
    Points = 1u << 4,
    Primvars = 1u << 5,
    Material = 1u << 6,
    Params = 1u << 7,
    Visibility = 1u << 8,
};

constexpr Dirty operator|(Dirty a, Dirty b) {
    return static_cast<Dirty>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
constexpr Dirty operator&(Dirty a, Dirty b) {
    return static_cast<Dirty>(static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
}
constexpr Dirty& operator|=(Dirty& a, Dirty b) {
    return a = a | b;
}
/// @return Whether `bits` has any of `flags`.
constexpr bool Any(Dirty bits, Dirty flags) {
    return (bits & flags) != Dirty::None;
}

/// @brief A limit of an attribute's numbers, inclusive or not.
struct Bound {
    double value = 0.0;
    bool exclusive = false;
};
constexpr Bound AtLeast(double value) {
    return {value, false};
}
constexpr Bound Above(double value) {
    return {value, true};
}
constexpr Bound AtMost(double value) {
    return {value, false};
}
constexpr Bound Below(double value) {
    return {value, true};
}

/// @brief What users need to author an attribute without reading code, and what it checks.
struct AttributeMeta {
    std::string_view doc{};
    std::string_view label{};
    std::string_view group{};
    /// A slider's range in editors; 0 and 0 for none.
    double min = 0.0;
    double max = 0.0;
    /// What `Set` accepts, per component.
    std::optional<Bound> lower{};
    std::optional<Bound> upper{};
    bool finite = false;
    /// The tokens a token attribute takes; any when empty.
    std::vector<std::string_view> options{};
    bool color = false;
    bool asset = false;
    /// A tensor of integers rather than floats.
    bool integer = false;
    /// A tensor of raw bytes, in elements of any type.
    bool bytes = false;
    /// A tensor's vector size: (n, components) or flat; 0 for a flat array.
    std::uint8_t components = 0;
    /// The ranks a tensor may have, as a dense grid's 3; none for flat or vector arrays.
    std::vector<std::int64_t> ranks{};
    Dirty dirty = Dirty::Params;
    /// Resolved from the nearest ancestor authoring it (other than as "inherited") into its
    /// `ResolvedName` column.
    bool inherited = false;
};

/// @brief One declared attribute.
struct AttributeDef {
    std::string_view name;
    ColumnType type = ColumnType::Double;
    Value fallback;
    AttributeMeta meta;
};

enum class SchemaKind : std::uint8_t {
    /// A base of other types, never a prim's own type.
    Abstract,
    /// A prim type.
    Concrete,
    /// An API schema applied once (`MaterialBindingAPI`).
    SingleApply,
    /// An API schema applied under instance names (`CollectionAPI:<name>`).
    MultipleApply,
};

class Scene;

/// @brief A bucket's prims as an extent function reads them: columns by attribute name, and the
/// scene for rules that read other prims.
struct ExtentRows {
    const PrimStore& store;
    BucketId bucket = 0;
    const Scene* scene = nullptr;

    /// @return The column of a declared attribute.
    template <typename T> std::span<const T> Column(std::string_view attribute) const {
        const std::optional<Token> token = store.Paths().Names().Find(attribute);
        return token ? store.Read<T>(bucket, *token) : std::span<const T>();
    }
    std::string_view Text(Token token) const {
        return store.Paths().Names().Text(token);
    }
    /// @return The live prim a relationship targets, or none.
    std::optional<PrimId> Resolve(const Target& target) const;
    /// @return Another prim's extent brought into a row's own space through the transforms
    /// between, or none when it has none.
    std::optional<Box> ExtentInRowSpace(PrimId other, std::uint32_t row) const;
};

/// Computes the extents of a bucket's rows [begin, end) into `out`, one box each, in each prim's
/// own space. Written as a loop over columns, as the GPU kernels are.
using ExtentFunction = void (*)(const ExtentRows& rows, RowRange range, std::span<Box> out);

/// @brief Fills `out` with `extent(row)` for each row of `range`: the loop every extent function
/// runs.
template <typename Extent>
void ComputeRows(RowRange range, std::span<Box> out, const Extent& extent) {
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        out[row - range.begin] = extent(row);
    }
}

/// The extent function of prims centred on their origin. `kHalves(rows)` reads the columns once
/// and returns each row's half-size by row.
template <auto kHalves>
void CentredExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto half = kHalves(rows);
    ComputeRows(range, out,
                [&](std::uint32_t row) { return Box::Around(glm::dvec3(0.0), half(row)); });
}

/// @brief How a prim type is bounded, as USD's Boundable and non-Boundable bases divide them.
struct ExtentRule {
    enum class Kind : std::uint8_t {
        /// The base's rule.
        Inherit,
        /// An extent from the prim's attributes.
        Computed,
        /// Bounds only through its descendants (Xform, Scope).
        Descendants,
        /// Infinite: left out of framing (distant and dome lights, skies).
        Unbounded,
        /// Not geometric (cameras, materials, render settings).
        None,
    };
    Kind kind = Kind::Inherit;
    ExtentFunction compute = nullptr;
    /// Reads other prims (a volume its fields), so it is computed afresh rather than cached.
    bool readsOtherPrims = false;
};
/// An extent from other prims' (a volume's from its fields).
constexpr ExtentRule ComputedFromOtherPrims(ExtentFunction compute) {
    return {ExtentRule::Kind::Computed, compute, true};
}
constexpr ExtentRule ComputedExtent(ExtentFunction compute) {
    return {ExtentRule::Kind::Computed, compute};
}
inline constexpr ExtentRule kInheritExtent{};
inline constexpr ExtentRule kDescendantsExtent{ExtentRule::Kind::Descendants};
inline constexpr ExtentRule kUnbounded{ExtentRule::Kind::Unbounded};
inline constexpr ExtentRule kNoExtent{ExtentRule::Kind::None};

/// @brief A prim type or API schema: its base, attributes and extent rule.
struct SchemaDef {
    std::string_view name;
    const SchemaDef* base = nullptr;
    SchemaKind kind = SchemaKind::Concrete;
    std::string_view doc;
    std::vector<AttributeDef> attributes;
    ExtentRule extent;
    /// Namespaces (`xformOp:`) or `*` whose undeclared attributes the type takes.
    std::vector<std::string_view> openNamespaces;
    /// API schemas every prim of the type has, as UsdLux's lights have `CollectionAPI:lightLink`.
    std::vector<std::string_view> builtinApis;
    /// Fallbacks the type changes for attributes of its bases or built-in schemas, as
    /// DistantLight's brighter intensity.
    std::vector<std::pair<std::string_view, Value>> fallbackOverrides;
    /// The prim types an API schema may be applied to, as MeshLightAPI's Mesh; any when empty.
    std::vector<std::string_view> appliesTo;

    bool IsA(const SchemaDef& other) const;
    bool IsOpen(std::string_view attribute) const;
    /// @return The attribute, declared here or by a base.
    const AttributeDef* Find(std::string_view attribute) const;
    /// @return Every attribute, the bases' first.
    std::vector<const AttributeDef*> AllAttributes() const;
    /// @return The built-in API schemas of the type and its bases.
    std::vector<std::string_view> AllBuiltinApis() const;
    /// @return The fallback the type or its nearest base gives an attribute in place of its
    /// declaration's, or null.
    const Value* FallbackOverride(std::string_view attribute) const;
    /// @return The type's fallback for an attribute: its own or a base's override, else the
    /// declaration's.
    const Value& Fallback(const AttributeDef& def) const {
        const Value* override = FallbackOverride(def.name);
        return override != nullptr ? *override : def.fallback;
    }
    /// @return The rule, the nearest base's when inherited.
    ExtentRule Extent() const;

    /// @return This type or the nearest base `test` accepts, or null.
    template <typename Test> const SchemaDef* FirstInLineage(const Test& test) const {
        for (const SchemaDef* type = this; type != nullptr; type = type->base) {
            if (test(*type)) {
                return type;
            }
        }
        return nullptr;
    }
};

/// @return Every registered schema, bases before the types deriving from them.
std::span<const SchemaDef* const> Schemas();
/// @return The schema named `name`, or null.
const SchemaDef* FindSchema(std::string_view name);
/// @return Every attribute declared inherited, once each.
std::span<const AttributeDef* const> InheritedAttributes();
/// @return The inherited attribute named `name`, or null.
const AttributeDef* FindInherited(std::string_view name);
/// @return The column holding an inherited attribute's resolved values:
/// `ernie:resolved:visibility`.
inline std::string ResolvedName(std::string_view attribute) {
    return std::string(tokens::kResolvedPrefix) + std::string(attribute);
}

/// What a multiple-apply schema's attribute names hold in place of the instance's name.
inline constexpr std::string_view kInstanceName = "__INSTANCE_NAME__";

/// @brief An applied API schema: `"MaterialBindingAPI"`, or `"CollectionAPI:lights"`.
struct ApiName {
    const SchemaDef* schema = nullptr;
    std::string_view instance;

    /// @throws std::invalid_argument for an unknown API schema or a missing or extra instance.
    static ApiName Parse(std::string_view api);
    /// @return A declared name with the instance's name in place of `kInstanceName`.
    std::string Attribute(std::string_view declared) const;
    /// @return The name `Parse` reads.
    std::string ToString() const;
    bool operator==(const ApiName&) const = default;
};

/// @brief Checks a value against a declared attribute's limits and options; the scene checks its
/// type as it converts it.
/// @param attribute The attribute's name, an instance's for a multiple-apply schema's.
/// @throws std::invalid_argument naming the prim, the attribute and the reason.
void Validate(const AttributeDef& def, std::string_view attribute, const Value& value,
              const Path& path);
/// @brief Checks an array for a declared tensor attribute: its element type and vector size.
/// @throws std::invalid_argument naming the prim, the attribute and the reason.
void ValidateArray(const AttributeDef& def, std::string_view attribute, const Tensor& array,
                   const Path& path);

/// @brief The tokens of an enum a token attribute takes, in enumerator order.
template <typename E> struct ChoiceTokens;

/// @brief A token attribute read and written as the enum `E` (`ChoiceTokens<E>`).
template <typename E> struct Choice {};

}  // namespace ernie

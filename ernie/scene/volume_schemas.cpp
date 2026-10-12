// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/volume_schemas.h"

#include "ernie/scene/schema.h"

#include <algorithm>

namespace ernie {

ERNIE_DEFINE_SCHEMA(FieldBase, tokens::kFieldBase,
                    &Imageable::Definition(), true,
                    "A field of values over space, which Volumes "
                    "name (UsdVol).",
                    ERNIE_FIELD_BASE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(FieldAsset, tokens::kFieldAsset,
                    &FieldBase::Definition(), true,
                    "A field read from a file (UsdVol).",
                    ERNIE_FIELD_ASSET_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(OpenVDBAsset, tokens::kOpenVDBAsset,
                    &FieldAsset::Definition(), false,
                    "A grid of a NanoVDB file (.nvdb), which the GPU "
                    "reads in place.",
                    ERNIE_OPENVDB_ASSET_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(NanoVDBField, tokens::kNanoVDBField,
                    &FieldBase::Definition(), false,
                    "NanoVDB grids written as an array (Warp's "
                    "Volume.array()), which the GPU reads in place.",
                    ERNIE_NANOVDB_FIELD_ATTRIBUTES)
ERNIE_DEFINE_OPEN_SCHEMA(
    Volume, tokens::kVolume, &Gprim::Definition(), false,
    "A participating medium: fields named by field:<name> "
    "relationships, shaded by its Material's volume inputs.",
    ERNIE_VOLUME_ATTRIBUTES, tokens::kField)

std::string FieldRelationship(std::string_view name) {
    return std::string(tokens::kField) + ":" + std::string(name);
}

Volume& Volume::SetField(std::string_view name, const Path& field) {
    Edit().Set(GetPath(), FieldRelationship(name), Value(field));
    return *this;
}

Path Volume::GetField(std::string_view name) const {
    return GetTarget(GetScene(), GetPath(), FieldRelationship(name));
}

bool Volume::HasField(std::string_view name) const {
    return Has(FieldRelationship(name));
}

Volume& Volume::ClearField(std::string_view name) {
    Edit().Clear(GetPath(), FieldRelationship(name));
    return *this;
}

std::vector<std::string> Volume::FieldNames() const {
    const std::string prefix = FieldRelationship("");
    std::vector<std::string> names;
    for (const std::string& attribute :
         GetScene().Attributes(GetPath())) {
        if (attribute.starts_with(prefix) &&
            attribute.find(':', prefix.size()) == std::string::npos) {
            names.push_back(attribute.substr(prefix.size()));
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::vector<const PrimTypeDef*> VolumeSchemaTypes() {
    return {&FieldBase::Definition(), &FieldAsset::Definition(),
            &OpenVDBAsset::Definition(), &NanoVDBField::Definition(),
            &Volume::Definition()};
}

}  // namespace ernie

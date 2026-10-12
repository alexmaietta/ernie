// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/volumes.h"

#include "ernie/scene/material_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/volume_schemas.h"
#include "ernie/session/resolve.h"

#include <glm/gtc/matrix_inverse.hpp>

#include <optional>
#include <string_view>

namespace ernie {

namespace {

std::optional<VolumeFieldParams> FieldOf(const Scene& scene,
                                         const Volume& volume,
                                         std::string_view name,
                                         const glm::dmat4& toLocal) {
    if (name.empty()) {
        return std::nullopt;
    }
    const Path target = volume.GetField(name);
    if (target.IsEmpty() || !scene.Has(target)) {
        return std::nullopt;
    }
    const PrimTypeDef* type = PrimTypeOf(scene, target);
    if (type == nullptr || !type->IsA(OpenVDBAsset::Definition())) {
        return std::nullopt;
    }
    const OpenVDBAsset asset(scene, target);
    VolumeFieldParams field;
    field.file = asset.GetFilePath();
    field.gridName = asset.GetFieldName();
    field.gridIndex = asset.GetFieldIndex();
    field.toVolume = toLocal * scene.WorldTransform(target);
    if (field.file.empty()) {
        return std::nullopt;
    }
    return field;
}

VolumeMaterialParams MaterialOf(const Scene& scene,
                                const Volume& volume) {
    VolumeMaterialParams params;
    const Path bound = volume.GetMaterialBinding();
    if (bound.IsEmpty() || !scene.Has(bound) ||
        scene.TypeOf(bound) != tokens::kMaterial) {
        return params;
    }
    const Material material(scene, bound);
    params.density = material.GetDensity();
    params.albedo = material.GetVolumeAlbedo();
    params.anisotropy = material.GetAnisotropy();
    params.emission =
        material.GetEmissionColor() * material.GetEmissionStrength();
    params.blackbody = material.GetBlackbody();
    params.temperature = material.GetTemperature();
    params.temperatureScale = material.GetTemperatureScale();
    return params;
}

}  // namespace

std::vector<VolumeParams> VolumesOf(const Scene& scene) {
    std::vector<VolumeParams> volumes;
    for (const Path& path : scene.Prims(tokens::kVolume)) {
        if (!scene.IsVisible(path)) {
            continue;
        }
        const Volume volume(scene, path);
        VolumeParams params;
        params.prim = path.String();
        params.transform = scene.WorldTransform(path);
        params.world = WorldOf(scene, path);
        params.material = MaterialOf(scene, volume);
        const glm::dmat4 toLocal = glm::inverse(params.transform);
        std::string densityField = "density";
        std::string temperatureField = "temperature";
        std::string emissionField;
        const Path bound = volume.GetMaterialBinding();
        if (!bound.IsEmpty() && scene.Has(bound) &&
            scene.TypeOf(bound) == tokens::kMaterial) {
            const Material material(scene, bound);
            densityField = material.GetDensityField();
            temperatureField = material.GetTemperatureField();
            emissionField = material.GetEmissionField();
        }
        params.density =
            FieldOf(scene, volume, densityField, toLocal);
        params.temperature =
            FieldOf(scene, volume, temperatureField, toLocal);
        params.emission =
            FieldOf(scene, volume, emissionField, toLocal);
        volumes.push_back(std::move(params));
    }
    return volumes;
}

}  // namespace ernie

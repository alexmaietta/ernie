// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/volumes.h"

#include "ernie/geometry/mesh_data.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/volume_schemas.h"
#include "ernie/session/resolve.h"
#include "ernie/util/profile.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <optional>
#include <string_view>
#include <utility>

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
    if (type != nullptr && type->IsA(DenseField::Definition())) {
        const DenseField dense(scene, target);
        const Tensor* values = dense.GetValues();
        if (values == nullptr || values->Elements() == 0) {
            // Its values may come from a writer on the GPU
            // (`RenderDelegate::SetFieldWriter`), which the renderer
            // looks for.
            VolumeFieldParams field;
            field.toVolume = toLocal * scene.WorldTransform(target);
            field.array = target.String();
            field.dense = true;
            field.voxelSize = dense.GetVoxelSize();
            return field;
        }
        if (values->Type() != DType::Float32 || values->Rank() != 3 ||
            (values->Location().kind != DeviceKind::Cpu &&
             !values->IsContiguous())) {
            throw std::invalid_argument(
                target.String() +
                ": a DenseField's values must be a float32 (x, y, z) "
                "array, contiguous on a device");
        }
        VolumeFieldParams field;
        field.toVolume = toLocal * scene.WorldTransform(target);
        field.array = target.String();
        field.buffer = *values;
        field.version =
            scene.AttributeVersion(target, tokens::kDenseValues);
        field.stream =
            scene.WriteStream(target, tokens::kDenseValues);
        field.dense = true;
        field.voxelSize = dense.GetVoxelSize();
        return field;
    }
    if (type != nullptr && type->IsA(NanoVDBField::Definition())) {
        const NanoVDBField grids(scene, target);
        const Tensor* buffer = grids.GetBuffer();
        if (buffer == nullptr || buffer->Elements() == 0) {
            return std::nullopt;
        }
        if (buffer->Location().kind != DeviceKind::Cpu &&
            !buffer->IsContiguous()) {
            throw std::invalid_argument(
                target.String() +
                ": NanoVDB grids on a device must be contiguous");
        }
        VolumeFieldParams field;
        field.stream =
            scene.WriteStream(target, tokens::kNanoVdbBuffer);
        field.gridName = grids.GetFieldName();
        field.gridIndex = grids.GetFieldIndex();
        field.toVolume = toLocal * scene.WorldTransform(target);
        field.array = target.String();
        field.buffer = *buffer;
        field.version =
            scene.AttributeVersion(target, tokens::kNanoVdbBuffer);
        return field;
    }
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
    params.secondAnisotropy = material.GetSecondAnisotropy();
    params.lobeMix = material.GetLobeMix();
    params.powder = material.GetPowder();
    params.powderLength = material.GetPowderLength();
    params.phase = material.GetPhaseFunction();
    // Hapke's b and c, as its BRDF reads them (with its defaults).
    params.b = GetNumber(scene, bound, tokens::kPhaseB, params.b);
    params.c = GetNumber(scene, bound, tokens::kPhaseC, params.c);
    params.dropletRadius = material.GetDropletRadius();
    params.dropletSpread = material.GetDropletSpread();
    params.surfaceDensity = material.GetSurfaceDensity();
    params.octaves = material.GetScatteringOctaves();
    params.octaveAttenuation = material.GetOctaveAttenuation();
    params.octaveExtinction = material.GetOctaveExtinction();
    params.octavePhase = material.GetOctavePhase();
    params.skyLight = material.GetSkyLight();
    params.emission =
        material.GetEmissionColor() * material.GetEmissionStrength();
    params.blackbody = material.GetBlackbody();
    params.temperature = material.GetTemperature();
    params.temperatureScale = material.GetTemperatureScale();
    return params;
}

VolumeParams VolumeOf(const Scene& scene, const Path& path) {
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
    params.density = FieldOf(scene, volume, densityField, toLocal);
    params.temperature =
        FieldOf(scene, volume, temperatureField, toLocal);
    params.emission = FieldOf(scene, volume, emissionField, toLocal);
    return params;
}

// Instance `i`'s transform of `data`: translate, then orient, then
// scale, as a PointInstancer's.
glm::dmat4 InstanceTransform(const InstanceData& data,
                             std::size_t i) {
    glm::dmat4 transform = glm::translate(
        glm::dmat4(1.0),
        glm::dvec3(data.positions[3 * i], data.positions[3 * i + 1],
                   data.positions[3 * i + 2]));
    if (4 * i + 3 < data.orientations.size()) {
        transform *= glm::mat4_cast(glm::dquat(
            data.orientations[4 * i + 3], data.orientations[4 * i],
            data.orientations[4 * i + 1],
            data.orientations[4 * i + 2]));
    }
    if (3 * i + 2 < data.scales.size()) {
        transform =
            glm::scale(transform, glm::dvec3(data.scales[3 * i],
                                             data.scales[3 * i + 1],
                                             data.scales[3 * i + 2]));
    }
    return transform;
}

}  // namespace

std::vector<VolumeParams> VolumesOf(const Scene& scene) {
    ERNIE_ZONE("Resolve volumes");
    // PointInstancers' prototypes, which show only as their
    // instances.
    const std::vector<Path> instancers =
        scene.Prims(tokens::kPointInstancer);
    std::vector<Path> prototypes;
    for (const Path& instancer : instancers) {
        for (const Path& prototype :
             PointInstancer(scene, instancer).GetPrototypes()) {
            prototypes.push_back(prototype);
        }
    }
    const auto isPrototype = [&prototypes](const Path& path) {
        return std::any_of(prototypes.begin(), prototypes.end(),
                           [&path](const Path& prototype) {
                               return path.HasPrefix(prototype);
                           });
    };
    std::vector<VolumeParams> volumes;
    const std::vector<Path> all = scene.Prims(tokens::kVolume);
    for (const Path& path : all) {
        if (scene.IsVisible(path) && !isPrototype(path)) {
            volumes.push_back(VolumeOf(scene, path));
        }
    }
    // Each instancer's volume prototypes, placed at its instances
    // (those with CPU arrays; GPU poses are not read yet).
    for (const Path& path : instancers) {
        const PointInstancer instancer(scene, path);
        const Tensor* positions = instancer.GetPositions();
        if (!scene.IsVisible(path) || positions == nullptr ||
            positions->Location().kind != DeviceKind::Cpu) {
            continue;
        }
        const std::vector<Path> targets = instancer.GetPrototypes();
        std::vector<std::pair<std::size_t, Path>> placed;
        for (std::size_t k = 0; k < targets.size(); ++k) {
            for (const Path& volume : all) {
                if (volume.HasPrefix(targets[k])) {
                    placed.emplace_back(k, volume);
                }
            }
        }
        if (placed.empty()) {
            continue;
        }
        const InstanceData data = InstanceDataOf(scene, path);
        const std::size_t count = data.positions.size() / 3;
        const std::vector<std::vector<std::uint32_t>> lists =
            InstancesByPrototype(scene, path, targets.size());
        const std::vector<std::int32_t> worlds =
            InstanceWorldsOf(scene, path, count);
        const glm::dmat4 world = scene.WorldTransform(path);
        for (const auto& [k, volumePath] : placed) {
            if (lists[k].empty()) {
                continue;
            }
            VolumeParams params = VolumeOf(scene, volumePath);
            // The volume below its prototype, the prototype in the
            // instance's frame.
            const glm::dmat4 inPrototype =
                scene.LocalTransform(targets[k]) *
                glm::inverse(scene.WorldTransform(targets[k])) *
                params.transform;
            for (const std::uint32_t i : lists[k]) {
                if (i >= count) {
                    continue;
                }
                params.instances.push_back(
                    world * InstanceTransform(data, i) * inPrototype);
                params.instanceWorlds.push_back(
                    i < worlds.size() ? worlds[i] : params.world);
            }
            params.prim = path.String();
            volumes.push_back(std::move(params));
        }
    }
    return volumes;
}

}  // namespace ernie

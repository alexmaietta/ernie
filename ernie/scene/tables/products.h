// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/device/tensor.h"
#include "ernie/scene/records/products.h"
#include "ernie/scene/tables/sensor_pipelines.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"
#include "ernie/stage/structure/structure.h"

namespace ernie {

class Scene;

enum class RenderEntry : uint32_t {
    MinimalData,
    MinimalDisplayColor,
    MinimalTextureDiffuse,
    MinimalFull,
    RealTime,
    RealTimeBands,
    Lidar,
    Radar,
};

inline constexpr uint32_t kRenderEntryCount = 8;

inline constexpr const char* kRenderEntryNames[kRenderEntryCount] = {"RenderMinimalData",
                                                                     "RenderMinimalDisplayColor",
                                                                     "RenderMinimalTextureDiffuse",
                                                                     "RenderMinimalFull",
                                                                     "RenderRealTime",
                                                                     "RenderRealTimeBands",
                                                                     "RenderLidar",
                                                                     "RenderRadar"};

enum class PointChannel : uint32_t {
    Coordinates,
    Distance,
    Intensity,
    StableId,
    Emitter,
    Firing,
    Echo,
    RadialVelocity,
    InstanceId,
    RadialVelocityVector,
};

struct PointChannelInfo {
    std::string_view name;
    device::DType type;
    int64_t lanes;
};

inline constexpr PointChannelInfo kPointChannels[] = {
    {"Coordinates", device::DType::Float32, 3},
    {"Distance", device::DType::Float32, 1},
    {"Intensity", device::DType::Float32, 1},
    {"StableId", device::DType::UInt32, 4},
    {"Emitter", device::DType::UInt32, 1},
    {"Firing", device::DType::UInt32, 1},
    {"Echo", device::DType::UInt32, 1},
    {"RadialVelocity", device::DType::Float32, 1},
    {"InstanceId", device::DType::Int64, 1},
    {"RadialVelocityVector", device::DType::Float32, 3}};

struct ResolvedVar {
    uint32_t prim = kNoPrim;
    std::string path;
    RenderVarSource source = RenderVarSource::LdrColor;
    FrameOfReference frame = FrameOfReference::Sensor;
    device::DType type = device::DType::UInt8;
    std::vector<int64_t> shape;
    std::vector<PointChannel> channels;
    std::string output;
    bool tap = false;
    bool semantic = false;

    bool operator==(const ResolvedVar&) const = default;
};

struct ProductShape {
    SchemaId sensor = SchemaId::None;
    std::vector<uint32_t> sensors;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t returns = 0;
    uint32_t channels = 0;
    float frameRate = 0;
    float startTime = 0;
    float missDistance = 0;
    std::vector<ResolvedVar> vars;
    std::vector<std::string> overlays;
    Token taxonomy;
    RenderEntry entry = RenderEntry::MinimalData;
    uint32_t pipeline = kNoPrim;
    std::vector<sensor_pipeline::SignalLayout> layouts;
    bool semantic = false;
    std::string error;

    bool operator==(const ProductShape&) const = default;
};

struct ResolvedProduct : ProductShape {
    uint32_t prim = kNoPrim;
    ProductRecord record{};
    uint64_t version = 0;
};

class ProductTable : public RenderTable {
  public:
    ProductTable(Scene& scene, Structure& structure, const SensorPipelineTable& pipelines)
        : m_scene(scene), m_structure(structure), m_pipelines(pipelines) {}

    std::string_view Name() const override {
        return "products";
    }

    std::span<const ColumnRef> Inputs() const override;

    void Run(const ChangeSet& changes, TableVersions& versions) override;
    void Describe(RenderTables& tables) const override;

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_records.Buffer()});
    }

    std::shared_ptr<const ResolvedProduct> Find(uint32_t prim) const;
    std::vector<std::shared_ptr<const ResolvedProduct>> Products() const;

  private:
    ResolvedProduct Resolve(uint32_t prim) const;
    void Resolve();

    Scene& m_scene;
    Structure& m_structure;
    const SensorPipelineTable& m_pipelines;
    uint64_t m_pipelineVersion = 0;
    mutable std::mutex m_mutex;
    std::map<uint32_t, std::shared_ptr<const ResolvedProduct>> m_products;
    RecordTable<ProductRecord> m_records{"product records"};
    uint64_t m_versions = 0;
    uint64_t m_seen = 0;
};

}

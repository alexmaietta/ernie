// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/device/tensor.h"
#include "ernie/scene/records/products.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"

namespace ernie {

class Scene;

struct SimulationClock {
    double start = 0;
    double end = 0;
    uint64_t steps = 0;
    uint64_t resets = 0;
};

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

enum class PointChannel : uint32_t {
    Coordinates,
    Distance,
    Intensity,
    StableId,
    Emitter,
    Firing,
    Echo,
    RadialVelocity,
};

struct PointChannelInfo {
    std::string_view name;
    device::DType type;
    int64_t lanes;
};

inline constexpr PointChannelInfo kPointChannels[] = {
    {"Coordinates", device::DType::Float32, 3}, {"Distance", device::DType::Float32, 1},
    {"Intensity", device::DType::Float32, 1},   {"StableId", device::DType::UInt32, 4},
    {"Emitter", device::DType::UInt32, 1},      {"Firing", device::DType::UInt32, 1},
    {"Echo", device::DType::UInt32, 1},         {"RadialVelocity", device::DType::Float32, 1}};

struct ResolvedVar {
    uint32_t prim = kNoPrim;
    std::string path;
    RenderVarSource source = RenderVarSource::LdrColor;
    FrameOfReference frame = FrameOfReference::Sensor;
    device::DType type = device::DType::UInt8;
    std::vector<int64_t> shape;
    std::vector<PointChannel> channels;
    bool semantic = false;

    bool operator==(const ResolvedVar&) const = default;
};

struct ResolvedProduct {
    uint32_t prim = kNoPrim;
    ProductRecord record{};
    std::vector<uint32_t> sensors;
    std::vector<ResolvedVar> vars;
    std::vector<std::string> overlays;
    Token taxonomy;
    RenderEntry entry = RenderEntry::MinimalData;
    bool semantic = false;
    std::string error;
    uint64_t version = 0;
    bool due = true;
};

struct PlannedCapture {
    uint32_t product = kNoPrim;
    uint32_t sensor = kNoPrim;
    uint32_t layer = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t taxonomy = 0;
    uint32_t returns = 1;
    float missDistance = 0;
    uint64_t firstPixel = 0;
};

struct ProductPlan {
    uint64_t id = 0;
    std::vector<std::string> paths;
    std::vector<ResolvedProduct> products;
    std::array<std::vector<PlannedCapture>, kRenderEntryCount> captures;
    std::array<uint64_t, kRenderEntryCount> pixelCounts{};
    uint32_t captureCount = 0;
    uint64_t pixelCount = 0;
    std::vector<Token> taxonomies;
};

inline constexpr size_t kCachedPlans = 4;

class ProductTable : public Table {
  public:
    ProductTable(Scene& scene, Structure& structure, const SimulationClock& clock)
        : m_scene(scene), m_structure(structure), m_clock(clock) {}

    std::string_view Name() const override {
        return "products";
    }

    std::span<const ColumnRef> Inputs() const override;

    bool Pending() const override {
        return m_clock.steps != m_scheduled;
    }

    void Run(const ChangeSet& changes, TableVersions& versions) override;
    void Describe(RenderTables& tables) const override;

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_records.Buffer()});
    }

    const ResolvedProduct* Find(uint32_t prim) const;
    std::shared_ptr<const ProductPlan> Plan(std::span<const std::string> paths);

  private:
    struct Schedule {
        float frameRate = 0;
        float startTime = 0;
        int64_t next = 0;
    };

    ResolvedProduct Resolve(uint32_t prim) const;
    void Resolve();
    void ScheduleCaptures();

    Scene& m_scene;
    Structure& m_structure;
    std::map<uint32_t, ResolvedProduct> m_products;
    RecordTable<ProductRecord> m_records{"product records"};
    uint64_t m_versions = 0;
    const SimulationClock& m_clock;
    std::map<uint32_t, Schedule> m_schedules;
    uint64_t m_scheduled = ~uint64_t(0);
    uint64_t m_resets = 0;
    uint64_t m_seen = 0;
    std::mutex m_planMutex;
    std::list<std::shared_ptr<const ProductPlan>> m_plans;
    uint64_t m_planIds = 0;
};

}

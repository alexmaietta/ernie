// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

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

struct ResolvedVar {
    uint32_t prim = kNoPrim;
    std::string path;
    RenderVarSource source = RenderVarSource::LdrColor;
    std::vector<std::string> channels;
    FrameOfReference frame = FrameOfReference::Sensor;

    bool operator==(const ResolvedVar&) const = default;
};

struct ResolvedProduct {
    uint32_t prim = kNoPrim;
    ProductRecord record{};
    std::vector<uint32_t> sensors;
    std::vector<ResolvedVar> vars;
    Token taxonomy;
    std::string error;
    uint64_t version = 0;
    bool due = true;
};

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
};

}

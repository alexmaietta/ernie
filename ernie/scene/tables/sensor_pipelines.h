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

#include "ernie/device/program.h"
#include "ernie/scene/tables/connections.h"
#include "ernie/scene/tables/pack.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"
#include "ernie/sensor_pipeline/definition.h"
#include "ernie/sensor_pipeline/node.h"
#include "ernie/stage/structure/structure.h"

namespace ernie {

inline constexpr std::string_view kTapSource = "Tap";

struct SensorPipelineInstance {
    uint32_t prim = kNoPrim;
    std::string path;
    uint32_t definition = 0;
    sensor_pipeline::ResolvedSensorPipeline resolved;
    std::vector<uint32_t> words;
    uint32_t firstWord = 0;
    std::string error;
};

class SensorPipelineTable : public RenderTable {
  public:
    explicit SensorPipelineTable(Structure& structure);

    std::string_view Name() const override {
        return "pipelines";
    }

    std::span<const ColumnRef> Inputs() const override;

    bool Pending() const override {
        return Registry() != sensor_pipeline::SensorPipelineNodes();
    }

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_words.Buffer()});
    }

    uint64_t WordsAddress() const {
        return m_words.Address();
    }

    std::vector<uint32_t> ReadWords(const SensorPipelineInstance& instance) const;

    void Run(const ChangeSet& changes, TableVersions& versions) override;

    std::shared_ptr<const SensorPipelineInstance> Find(uint32_t prim) const;
    std::shared_ptr<const sensor_pipeline::SensorPipelineSet> Definitions() const;
    std::shared_ptr<const sensor_pipeline::NodeRegistry> Registry() const;

    uint64_t Version() const {
        return m_version;
    }

  private:
    struct Gathered {
        uint32_t parameter = 0;
        ColumnRef column;
        uint32_t row = 0;
    };

    void Pack(bool rewrite);

    struct Tap {
        std::string name;
        uint32_t node = kNoPrim;
        std::string output;
    };

    std::vector<Tap> GatherTaps() const;
    sensor_pipeline::SensorPipelineDesc Gather(uint32_t pipeline, std::span<const Tap> taps) const;
    sensor_pipeline::NodeDesc GatherNode(uint32_t prim) const;
    std::vector<sensor_pipeline::ConnectionDesc> Connections(uint32_t prim) const;

    Structure& m_structure;
    sensor_pipeline::SensorPipelineLibrary m_library;
    std::shared_ptr<const sensor_pipeline::NodeRegistry> m_registry;
    mutable std::map<uint32_t, std::vector<Gathered>> m_gathered;
    ConnectionCache m_connections{m_structure, {SchemaId::SensorNode, SchemaId::SensorPipeline}};
    PackPlan m_pack{"pipeline"};
    DeviceTable<uint32_t> m_words{"pipeline words"};
    mutable std::vector<ColumnRef> m_inputs;
    mutable uint32_t m_inputSchemas = 0;
    mutable std::mutex m_mutex;
    std::map<uint32_t, std::shared_ptr<const SensorPipelineInstance>> m_instances;
    uint64_t m_seen = 0;
    uint64_t m_version = 0;
};

}

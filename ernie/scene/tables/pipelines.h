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

#include "ernie/pipeline/builtins.h"
#include "ernie/pipeline/definition.h"
#include "ernie/scene/tables/table.h"
#include "ernie/stage/structure/structure.h"

namespace ernie {

inline constexpr std::string_view kTapSource = "Tap";

struct PipelineInstance {
    uint32_t prim = kNoPrim;
    std::string path;
    uint32_t definition = 0;
    pipeline::ResolvedPipeline resolved;
    std::vector<uint32_t> words;
    std::string error;
};

class PipelineTable : public RenderTable {
  public:
    explicit PipelineTable(Structure& structure);

    std::string_view Name() const override {
        return "pipelines";
    }

    std::span<const ColumnRef> Inputs() const override;

    bool Pending() const override {
        return Registry() != pipeline::PipelineNodes();
    }

    void Run(const ChangeSet& changes, TableVersions& versions) override;

    std::shared_ptr<const PipelineInstance> Find(uint32_t prim) const;
    std::shared_ptr<const pipeline::PipelineSet> Definitions() const;
    std::shared_ptr<const pipeline::NodeRegistry> Registry() const;

    uint64_t Version() const {
        return m_version;
    }

  private:
    struct Tap {
        std::string name;
        uint32_t node = kNoPrim;
        std::string output;
    };

    std::vector<Tap> GatherTaps() const;
    pipeline::PipelineDesc Gather(uint32_t pipeline, std::span<const Tap> taps) const;
    pipeline::NodeDesc GatherNode(uint32_t prim) const;
    std::vector<pipeline::ConnectionDesc> Connections(uint32_t prim) const;

    Structure& m_structure;
    pipeline::PipelineLibrary m_library;
    std::shared_ptr<const pipeline::NodeRegistry> m_registry;
    mutable std::vector<ColumnRef> m_inputs;
    mutable uint32_t m_inputSchemas = 0;
    mutable std::mutex m_mutex;
    std::map<uint32_t, std::shared_ptr<const PipelineInstance>> m_instances;
    uint64_t m_seen = 0;
    uint64_t m_version = 0;
};

}

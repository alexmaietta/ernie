// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/commands.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/records/transforms.h"
#include "ernie/scene/tables/kernels/transforms.h"
#include "ernie/scene/tables/regions.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"
#include "ernie/stage/structure/structure.h"

namespace ernie {

struct SimulationClock;

class TransformTable : public RenderTable {
  public:
    explicit TransformTable(Structure& structure, const SimulationClock* clock = nullptr);

    std::string_view Name() const override {
        return "hierarchy";
    }

    std::span<const ColumnRef> Inputs() const override;
    std::span<const TableId> TableOutputs() const override;
    void Describe(RenderTables& tables) const override;
    void Hold(std::vector<Ref<const RefCounted>>& held) const override;
    void Prepare() override;
    void Run(const ChangeSet& changes, TableVersions& versions) override;

    bool Pending() const override {
        return m_follows.lagged;
    }

    uint32_t Followers() const {
        return uint32_t(m_follows.frames.size());
    }

    uint32_t Segments() const {
        return m_segments.Size();
    }

    uint32_t Dispatched() const {
        return m_dispatched;
    }

    uint32_t Rebuilds() const {
        return m_rebuilds;
    }

    const Ref<device::Buffer>& Transforms() const {
        return m_transforms.Buffer();
    }

    const Ref<device::Buffer>& Imageables() const {
        return m_imageables.Buffer();
    }

  private:
    struct FollowPlan {
        std::vector<uint32_t> frames;
        std::vector<uint32_t> frameStarts;
        std::vector<uint32_t> owned;
        std::vector<uint32_t> owners;
        std::vector<uint32_t> ownedStarts;
        std::vector<uint32_t> worlds;
        bool lagged = false;
    };

    void Rebuild();
    void PlanFollows();
    void RecordFollows(device::CommandEncoder& encoder, const ResolveParams& resolved);
    void WriteMotions(bool restructured);
    bool Apply(const ChangeSet& changes);
    uint32_t ParentRow(uint32_t prim) const;

    Structure& m_structure;
    const SimulationClock* m_clock = nullptr;
    device::Program& Kernels();

    Ref<device::Program> m_program;
    RecordTable<uint32_t> m_parents{"hierarchy parents"};
    RecordTable<uint32_t> m_xforms{"hierarchy transforms"};
    std::unordered_map<uint32_t, uint32_t> m_rows;
    Regions m_segments;
    Ref<device::Buffer> m_segmentRows;
    Ref<device::Buffer> m_segmentStart;
    Ref<device::Buffer> m_segmentCount;
    DeviceTable<WorldTransform> m_transforms{"transforms"};
    DeviceTable<ResolvedImageable> m_imageables{"imageables"};
    DeviceTable<MotionRecord> m_motions{"motion records"};
    uint64_t m_motionSeen = 0;
    FollowPlan m_follows;
    Ref<device::Buffer> m_followRows;
    Ref<device::Buffer> m_followOwners;
    DeviceTable<FollowState> m_followStates{"follow states"};
    uint64_t m_followSeen = 0;
    uint64_t m_clockSteps = 0;
    uint64_t m_clockResets = 0;
    double m_clockEnd = 0;
    bool m_built = false;
    uint64_t m_seen = 0;
    uint32_t m_dispatched = 0;
    uint32_t m_rebuilds = 0;
};

}

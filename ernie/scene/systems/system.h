// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/ref.h"
#include "ernie/scene/kernels/schema/schema.h"
#include "ernie/scene/timeline/table_log.h"

namespace ernie {

struct ChangeSet;
struct RenderTables;

class System {
  public:
    virtual ~System() = default;

    virtual std::string_view Name() const = 0;
    virtual std::span<const ColumnRef> Inputs() const = 0;

    virtual std::span<const TableId> TableInputs() const {
        return {};
    }

    virtual std::span<const TableId> TableOutputs() const {
        return {};
    }

    virtual bool Structural() const {
        return true;
    }

    virtual bool Pending() const {
        return false;
    }

    virtual void Prepare() {}

    virtual void Finish() {}

    virtual void Describe(RenderTables&) const {}

    virtual void Hold(std::vector<Ref<const RefCounted>>&) const {}

    virtual void Run(const ChangeSet& changes, TableLog& log) = 0;

    uint32_t Runs() const {
        return m_runs;
    }

  private:
    friend class Timeline;

    uint32_t m_runs = 0;
};

inline void HoldBuffers(std::vector<Ref<const RefCounted>>& held,
                        std::initializer_list<Ref<device::Buffer>> buffers) {
    for (const Ref<device::Buffer>& buffer : buffers) {
        if (buffer) {
            held.emplace_back(buffer);
        }
    }
}

}

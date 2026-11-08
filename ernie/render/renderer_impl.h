// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "ernie/algorithms/scan.h"
#include "ernie/device/buffer.h"
#include "ernie/device/program.h"
#include "ernie/device/queue.h"
#include "ernie/overlay/overlay_impl.h"
#include "ernie/render/kernels/render/render.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tables/products.h"

namespace ernie {

class RendererImpl {
  public:
    explicit RendererImpl(Ref<Scene> scene, const RendererDesc& desc = {});

    RenderOutput Step(std::span<const std::string> products, double dt,
                      Stream stream = Stream::None());

    struct OwnedVar {
        uint32_t product = 0;
        std::string renderVar;
        device::Tensor tensor;
        std::string channel;
    };

    struct OwnedCloud {
        uint32_t product = kNoPrim;
        uint32_t slots = 0;
        Ref<device::Buffer> cloud;
        Ref<device::Buffer> records;
        Ref<device::Buffer> flags;
        Ref<device::Buffer> offsets;
    };

    struct OwnedOverlay {
        uint32_t product = 0;
        device::Tensor tensor;
        uint64_t depth = 0;
        Ref<device::Buffer> ownDepth;
        Ref<device::Buffer> primIds;
    };

    struct OutputSet {
        std::array<std::vector<Capture>, kRenderEntryCount> captures;
        std::array<Ref<device::Buffer>, kRenderEntryCount> captureBuffers;
        std::vector<OwnedVar> vars;
        std::vector<OwnedOverlay> overlays;
        std::vector<OwnedCloud> clouds;
        bool outlined = false;
    };

    struct StepPlan {
        std::shared_ptr<const ProductPlan> plan;
        std::vector<OutputSet> sets;
    };

    StepPlan& PlanProducts(std::span<const std::string> products);

    OutputSet& AcquireSet(StepPlan& plan);
    uint64_t AcquireCloud(OutputSet& set, uint32_t index, const std::string& path,
                          const ResolvedProduct& product, const ResolvedVar& var, uint64_t next);
    device::Operation Trace(const ProductPlan& plan, const OutputSet& set,
                            const std::set<uint32_t>& due, const Snapshot& snapshot);
    std::vector<OverlayContents> RecordOverlays(device::CommandEncoder& encoder,
                                                const ProductPlan& plan, const OutputSet& set,
                                                const std::set<uint32_t>& due,
                                                const Snapshot& snapshot);

    Ref<Scene> m_scene;
    uint64_t m_maxPixelsPerLaunch;
    Ref<Overlays> m_overlays;
    Ref<device::Program> m_renderProgram;

    std::unique_ptr<algorithms::Scan> m_scan;
    uint32_t m_scanCapacity = 0;
    std::list<StepPlan> m_plans;

    std::mutex m_stepMutex;
    std::optional<OverlayPass> m_overlayPass;
    std::vector<std::shared_ptr<OverlayPickState>> m_servedPicks;
};

}

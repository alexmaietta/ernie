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

inline constexpr size_t kCachedPlans = 4;

class RendererImpl {
  public:
    explicit RendererImpl(Ref<Scene> scene, const RendererDesc& desc = {});

    RenderOutput Step(std::span<const std::string> products, double dt,
                      Stream stream = Stream::None());

    struct Product : ResolvedProduct {
        uint32_t mapVersion = ~0u;
        std::vector<RenderVarOutput> outputs;
    };

    struct Launch {
        std::vector<Capture> captures;
        uint64_t pixelCount = 0;
    };

    struct OwnedVar {
        std::string product;
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
        std::string product;
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
        std::array<Launch, kRenderEntryCount> launches;
        uint32_t captureCount = 0;
        uint64_t pixelCount = 0;
        std::vector<Token> taxonomies;
        std::map<std::string, ProductOutput, std::less<>> products;
        std::map<std::string, std::vector<std::string>, std::less<>> shownOverlays;
        std::vector<OutputSet> sets;
    };

    void DropStaleProducts();
    StepPlan& PlanProducts(std::span<const std::string> products);
    Product& PrepareProduct(const std::string& path);
    void UpdateMaps();

    OutputSet& AcquireSet(StepPlan& plan);
    uint64_t AcquireCloud(OutputSet& set, const std::string& path, const Product& product,
                          const ResolvedVar& var, uint64_t next);
    device::Operation Trace(const OutputSet& set, const std::set<uint32_t>& due,
                            const Snapshot& snapshot);
    std::vector<OverlayContents> RecordOverlays(device::CommandEncoder& encoder,
                                                const OutputSet& set, const std::set<uint32_t>& due,
                                                const Snapshot& snapshot);

    void SetOverlay(std::string_view name, Overlay overlay);
    void RemoveOverlay(std::string_view name);
    std::vector<std::string> OverlayNames() const;
    void ShowOverlays(std::string_view product, std::span<const std::string> names);
    void HideOverlays(std::string_view product);
    OverlayPick PickOverlay(std::string_view product, Float2 pixel, float radius, uint32_t layer);

    Ref<Scene> m_scene;
    uint64_t m_maxPixelsPerLaunch;
    Ref<device::Program> m_renderProgram;

    std::map<std::string, Product> m_products;
    std::unique_ptr<algorithms::Scan> m_scan;
    uint32_t m_scanCapacity = 0;
    std::list<StepPlan> m_plans;
    StepPlan* m_plan = nullptr;

    std::mutex m_stepMutex;
    mutable std::mutex m_overlayMutex;
    std::map<std::string, OverlayContents, std::less<>> m_overlays;
    std::map<std::string, std::vector<std::string>, std::less<>> m_shownOverlays;
    std::optional<OverlayPass> m_overlayPass;
    std::vector<std::shared_ptr<OverlayPickState>> m_picks;
    std::vector<std::shared_ptr<OverlayPickState>> m_servedPicks;
};

}

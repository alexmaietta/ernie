// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/plans.h"

#include <algorithm>
#include <optional>

#include "ernie/scene/scene.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"

namespace ernie {

std::shared_ptr<const ProductPlan> ProductPlans::Plan(std::span<const std::string> paths) {
    std::vector<std::string> sorted(paths.begin(), paths.end());
    std::ranges::sort(sorted);
    sorted.erase(std::ranges::unique(sorted).begin(), sorted.end());
    const std::lock_guard lock(m_mutex);
    const auto cached =
        std::ranges::find_if(m_plans, [&](const auto& plan) { return plan->paths == sorted; });
    if (cached != m_plans.end()) {
        const bool current = std::ranges::all_of((*cached)->products, [&](const auto& product) {
            const std::shared_ptr<const ResolvedProduct> now = m_products.Find(product->prim);
            return now && now->version == product->version;
        });
        if (current) {
            m_plans.splice(m_plans.begin(), m_plans, cached);
            return m_plans.front();
        }
        m_plans.erase(cached);
    }

    ERNIE_ZONE("Products/Plan");
    auto plan = std::make_shared<ProductPlan>();
    plan->id = ++m_ids;
    for (const std::string& path : sorted) {
        const std::optional<Prim> prim = m_scene.Find(path);
        if (!prim || !prim->IsA<RenderProduct>()) {
            throw Error("\"{}\" isn't a RenderProduct", path);
        }
        std::shared_ptr<const ResolvedProduct> product = m_products.Find(prim->Index());
        if (!product) {
            throw Error("{} isn't published yet", path);
        }
        if (!product->error.empty()) {
            throw Error("{}", product->error);
        }
        PlannedCapture capture{.product = product->prim,
                               .width = product->width,
                               .height = product->height,
                               .taxonomy = 0,
                               .missDistance = product->missDistance,
                               .returns = std::max(product->returns, 1u)};
        if (product->semantic) {
            const auto found = std::ranges::find(plan->taxonomies, product->taxonomy);
            capture.taxonomy = uint32_t(found - plan->taxonomies.begin());
            if (found == plan->taxonomies.end()) {
                plan->taxonomies.push_back(product->taxonomy);
            }
        }
        const auto entry = uint32_t(product->entry);
        uint64_t& pixels = plan->pixelCounts[entry];
        for (capture.layer = 0; capture.layer < product->sensors.size(); ++capture.layer) {
            capture.sensor = product->sensors[capture.layer];
            capture.firstPixel = pixels;
            plan->captures[entry].push_back(capture);
            pixels += uint64_t(capture.width) * capture.height;
        }
        const auto index = uint32_t(plan->products.size());
        plan->products.push_back(product);
        if (product->pipeline != kNoPrim) {
            for (const uint32_t sensor : product->sensors) {
                const auto found =
                    std::ranges::find_if(plan->pipelines, [&](const PlannedSensorPipeline& run) {
                        const ResolvedProduct& first = *plan->products[run.products.front()];
                        return run.pipeline == product->pipeline && run.sensor == sensor &&
                               first.width == product->width && first.height == product->height;
                    });
                PlannedSensorPipeline& run =
                    found != plan->pipelines.end()
                        ? *found
                        : plan->pipelines.emplace_back(
                              PlannedSensorPipeline{product->pipeline, sensor, {}});
                run.products.push_back(index);
            }
        }
        plan->captureCount += uint32_t(product->sensors.size());
        plan->pixelCount += uint64_t(product->width) * product->height * product->sensors.size();
    }
    plan->paths = std::move(sorted);
    m_plans.push_front(plan);
    if (m_plans.size() > kCachedPlans) {
        m_plans.pop_back();
    }
    return plan;
}

}

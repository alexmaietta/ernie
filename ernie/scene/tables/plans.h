// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "ernie/scene/records/products.h"
#include "ernie/scene/tables/products.h"

namespace ernie {

struct ProductPlan {
    uint64_t id = 0;
    std::vector<std::string> paths;
    std::vector<std::shared_ptr<const ResolvedProduct>> products;
    std::array<std::vector<PlannedCapture>, kRenderEntryCount> captures;
    std::array<uint64_t, kRenderEntryCount> pixelCounts{};
    uint32_t captureCount = 0;
    uint64_t pixelCount = 0;
    std::vector<Token> taxonomies;
};

inline constexpr size_t kCachedPlans = 4;

class ProductPlans {
  public:
    ProductPlans(Scene& scene, const ProductTable& products)
        : m_scene(scene), m_products(products) {}

    std::shared_ptr<const ProductPlan> Plan(std::span<const std::string> paths);

  private:
    Scene& m_scene;
    const ProductTable& m_products;
    std::mutex m_mutex;
    std::list<std::shared_ptr<const ProductPlan>> m_plans;
    uint64_t m_ids = 0;
};

}

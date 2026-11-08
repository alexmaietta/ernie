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
};

class ProductTable : public Table {
  public:
    ProductTable(Scene& scene, Structure& structure) : m_scene(scene), m_structure(structure) {}

    std::string_view Name() const override {
        return "products";
    }

    std::span<const ColumnRef> Inputs() const override;
    void Run(const ChangeSet& changes, TableVersions& versions) override;
    void Describe(RenderTables& tables) const override;

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_records.Buffer()});
    }

    const ResolvedProduct* Find(uint32_t prim) const;

  private:
    ResolvedProduct Resolve(uint32_t prim) const;

    Scene& m_scene;
    Structure& m_structure;
    std::map<uint32_t, ResolvedProduct> m_products;
    RecordTable<ProductRecord> m_records{"product records"};
    uint64_t m_versions = 0;
};

}

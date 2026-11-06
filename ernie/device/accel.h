// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "ernie/device/buffer.h"
#include "ernie/device/device.h"
#include "ernie/device/resource.h"
#include "ernie/export.h"

namespace ernie::device {

struct AccelTriangles {
    BufferRange vertices;
    uint32_t stride = 12;
    uint32_t vertexCount = 0;
    std::optional<BufferRange> indices;
    uint32_t count = 0;
    bool refittable = false;
};

struct AccelBoxes {
    BufferRange boxes;
    uint32_t count = 0;
    bool refittable = false;
};

struct AccelInstances {
    BufferRange instances;
    uint32_t count = 0;
};

enum class AccelLevel : uint8_t {
    Bottom,
    Top,
};

inline AccelLevel GetLevel(const AccelInput& input) {
    return std::holds_alternative<AccelInstances>(input) ? AccelLevel::Top : AccelLevel::Bottom;
}

inline bool Refittable(const AccelInput& input) {
    if (const auto* triangles = std::get_if<AccelTriangles>(&input)) {
        return triangles->refittable;
    }
    if (const auto* boxes = std::get_if<AccelBoxes>(&input)) {
        return boxes->refittable;
    }
    return true;
}

template <class Visit> void ForEachBuffer(const AccelTriangles& triangles, Visit visit) {
    visit(triangles.vertices);
    if (triangles.indices) {
        visit(*triangles.indices);
    }
}

template <class Visit> void ForEachBuffer(const AccelBoxes& boxes, Visit visit) {
    visit(boxes.boxes);
}

template <class Visit> void ForEachBuffer(const AccelInstances& instances, Visit visit) {
    visit(instances.instances);
}

template <class... Inputs, class Visit>
void ForEachBuffer(const std::variant<Inputs...>& input, Visit visit) {
    std::visit([&](const auto& held) { ForEachBuffer(held, visit); }, input);
}

struct AccelSizes {
    uint64_t storage = 0;
    uint64_t scratch = 0;
    uint64_t updateScratch = 0;
};

class ERNIE_API Accel : public Resource {
  public:
    ~Accel() override = default;

    uint64_t Address() const {
        return m_address;
    }

    Buffer& Storage() const {
        return *m_storage;
    }

    AccelLevel Level() const {
        return m_level;
    }

    const AccelSizes& Sizes() const {
        return m_sizes;
    }

  protected:
    Accel(Device& device, Ref<Buffer> storage, AccelLevel level, AccelSizes sizes,
          std::string label)
        : Resource(device, std::move(label)), m_storage(std::move(storage)), m_level(level),
          m_sizes(sizes) {}

    uint64_t m_address = 0;

  private:
    Ref<Buffer> m_storage;
    AccelLevel m_level;
    AccelSizes m_sizes;
};

}

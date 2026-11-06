// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>
#include <vector>

#include "ernie/bvh/kernels/bvh.h"
#include "ernie/bvh/reference.h"

namespace ernie::test {

inline std::vector<bvh::Triangle> Soup(uint32_t count, uint32_t seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> place(0, 1);
    std::uniform_real_distribution<float> offset(-0.02f, 0.02f);
    std::vector<bvh::Triangle> triangles(count);
    for (bvh::Triangle& triangle : triangles) {
        const float center[3] = {place(random), place(random), place(random)};
        for (float* corner : {triangle.a, triangle.b, triangle.c}) {
            for (int i = 0; i < 3; ++i) {
                corner[i] = center[i] + offset(random);
            }
        }
    }
    return triangles;
}

inline std::vector<bvh::Triangle> Grid(uint32_t size) {
    std::vector<bvh::Triangle> triangles;
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
            const float x0 = float(x), x1 = float(x + 1), y0 = float(y), y1 = float(y + 1);
            triangles.push_back({{x0, y0, 0}, {x1, y0, 0}, {x1, y1, 0}});
            triangles.push_back({{x0, y0, 0}, {x1, y1, 0}, {x0, y1, 0}});
        }
    }
    return triangles;
}

inline std::vector<bvh::Triangle> Points(uint32_t count) {
    return std::vector<bvh::Triangle>(count,
                                      {{0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}});
}

inline std::vector<BvhBox> Boxes(uint32_t count, uint32_t seed) {
    std::vector<BvhBox> boxes;
    for (const bvh::Triangle& triangle : Soup(count, seed)) {
        boxes.push_back(BvhTriangleBox(triangle));
    }
    return boxes;
}

struct Instances {
    std::vector<InstanceRecord> records;
    std::vector<BvhBox> bounds;

    std::vector<BvhBox> WorldBoxes() const {
        std::vector<BvhBox> boxes;
        for (size_t i = 0; i < records.size(); ++i) {
            boxes.push_back(BvhInstanceBox(records[i], bounds[i]));
        }
        return boxes;
    }
};

inline Instances Scatter(uint32_t count, uint32_t seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> place(0, 10);
    std::uniform_real_distribution<float> scale(0.2f, 1);
    Instances instances;
    for (uint32_t i = 0; i < count; ++i) {
        InstanceRecord record{};
        const float c = std::cos(float(i)), s = std::sin(float(i));
        const float rotation[3][3] = {{c, -s, 0}, {s, c, 0}, {0, 0, 1}};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                record.transform[row * 4 + column] = rotation[row][column] * scale(random);
            }
            record.transform[row * 4 + 3] = place(random);
        }
        record.custom = (i * 7 & kInstanceCustomMask) | 0xFFu << 24;
        record.blas = 0x10000 + uint64_t(i % 5) * 0x1000;
        BvhBox bounds;
        for (int axis = 0; axis < 3; ++axis) {
            bounds.lower[axis] = -scale(random);
            bounds.upper[axis] = scale(random);
        }
        instances.records.push_back(record);
        instances.bounds.push_back(bounds);
    }
    return instances;
}

inline BvhHeader Header(std::span<const std::byte> bytes) {
    BvhHeader header;
    std::memcpy(&header, bytes.data(), sizeof(header));
    return header;
}

inline BvhBox Bounds(std::span<const std::byte> bytes) {
    const BvhHeader header = Header(bytes);
    BvhBox bounds;
    std::memcpy(bounds.lower, header.lower, sizeof(bounds.lower));
    std::memcpy(bounds.upper, header.upper, sizeof(bounds.upper));
    return bounds;
}

}

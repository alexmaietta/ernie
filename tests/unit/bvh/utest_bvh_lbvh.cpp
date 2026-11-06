// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/bvh/lbvh.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/bvh/common.h"
#include "ernie/bvh/inspect.h"
#include "ernie/util/error.h"
#include "support/buffers.h"
#include "support/devices.h"
#include "support/geometry.h"

namespace ernie::bvh {
namespace {

using device::Buffer;
using device::BufferRange;
using device::CommandEncoder;
using device::Device;
using test::ForEachDevice;
using test::Upload;
using Bytes = std::vector<std::byte>;

struct Inputs {
    std::vector<Ref<Buffer>> buffers;
    std::vector<Geometry> geometries;

    void Add(Device& device, const std::vector<Triangle>& mesh) {
        buffers.push_back(Upload(device, mesh));
        geometries.push_back(device::AccelTriangles{.vertices = *buffers.back(),
                                                    .vertexCount = uint32_t(mesh.size() * 3),
                                                    .count = uint32_t(mesh.size())});
    }
};

std::vector<Bytes> Build(Device& device, const LbvhBuilder& builder,
                         std::span<const Geometry> geometries) {
    std::vector<Ref<Buffer>> outputs;
    std::vector<BufferRange> ranges;
    for (const Geometry& geometry : geometries) {
        const uint32_t count = std::visit([](const auto& held) { return held.count; }, geometry);
        outputs.push_back(device.CreateBuffer({.bytes = StorageBytes(count, Kind(geometry))}));
        ranges.emplace_back(*outputs.back());
    }
    CommandEncoder encoder = device.GetQueue().CreateEncoder();
    builder.Build(encoder, geometries, ranges);
    device.GetQueue().Submit(encoder.Finish());

    std::vector<Bytes> bvhs;
    for (const BufferRange& range : ranges) {
        bvhs.push_back(test::ReadBytes(device, range));
    }
    return bvhs;
}

BvhBox MeshBounds(const std::vector<Triangle>& mesh) {
    BvhBox bounds = BvhEmptyBox();
    for (const Triangle& triangle : mesh) {
        bounds = BvhUnion(bounds, BvhTriangleBox(triangle));
    }
    return bounds;
}

void ExpectWellFormed(const Bytes& bvh, uint32_t records, std::span<const BvhBox> boxes = {}) {
    const Inspection found = Inspect(bvh, boxes);
    ASSERT_TRUE(found.errors.empty()) << found.errors.front();
    EXPECT_EQ(found.records, records);
}

void ExpectBounds(const Bytes& bvh, const BvhBox& expected) {
    const BvhBox bounds = test::Bounds(bvh);
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_EQ(bounds.lower[axis], expected.lower[axis]) << axis;
        EXPECT_EQ(bounds.upper[axis], expected.upper[axis]) << axis;
    }
}

TEST(LbvhTest, BuildsMeshesOfEveryShapeTogether) {
    const std::vector<std::vector<Triangle>> meshes = {
        test::Soup(1, 1),    test::Soup(3, 2), test::Soup(4, 3), test::Soup(0, 4),
        test::Soup(1000, 5), test::Grid(10),   test::Points(50)};
    ForEachDevice([&](Device& device) {
        Inputs inputs;
        uint32_t total = 0;
        for (const std::vector<Triangle>& mesh : meshes) {
            inputs.Add(device, mesh);
            total += uint32_t(mesh.size());
        }
        const LbvhBuilder builder(device, total, uint32_t(meshes.size()));
        const std::vector<Bytes> bvhs = Build(device, builder, inputs.geometries);
        for (size_t m = 0; m < meshes.size(); ++m) {
            SCOPED_TRACE(m);
            ExpectWellFormed(bvhs[m], uint32_t(meshes[m].size()));
            if (!meshes[m].empty()) {
                ExpectBounds(bvhs[m], MeshBounds(meshes[m]));
            }
        }
    });
}

TEST(LbvhTest, BuildsOneMeshWithNarrowKeys) {
    const std::vector<Triangle> mesh = test::Soup(5000, 6);
    ForEachDevice([&](Device& device) {
        Inputs inputs;
        inputs.Add(device, mesh);
        const LbvhBuilder builder(device, uint32_t(mesh.size()));
        const Bytes bvh = Build(device, builder, inputs.geometries).front();
        ExpectWellFormed(bvh, uint32_t(mesh.size()));
        ExpectBounds(bvh, MeshBounds(mesh));
        EXPECT_LT(Inspect(bvh).sahCost, Inspect(BuildReference(mesh)).sahCost * 1.5);
    });
}

TEST(LbvhTest, ReadsIndexedVerticesAtTheirStride) {
    const std::vector<Triangle> mesh = test::Soup(300, 7);
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    for (uint32_t t = 0; t < mesh.size(); ++t) {
        for (const float* corner : {mesh[t].c, mesh[t].b, mesh[t].a}) {
            vertices.insert(vertices.end(), {corner[0], corner[1], corner[2], -1e9f});
        }
        indices.insert(indices.end(), {t * 3 + 2, t * 3 + 1, t * 3});
    }
    ForEachDevice([&](Device& device) {
        const Ref<Buffer> vertexBuffer = Upload(device, vertices);
        const Ref<Buffer> indexBuffer = Upload(device, indices);
        const std::vector<Geometry> geometries = {
            device::AccelTriangles{.vertices = *vertexBuffer,
                                   .stride = 16,
                                   .vertexCount = uint32_t(mesh.size() * 3),
                                   .indices = *indexBuffer,
                                   .count = uint32_t(mesh.size())}};
        const LbvhBuilder builder(device, uint32_t(mesh.size()));
        const Bytes bvh = Build(device, builder, geometries).front();
        ExpectWellFormed(bvh, uint32_t(mesh.size()));
        ExpectBounds(bvh, MeshBounds(mesh));
    });
}

TEST(LbvhTest, BuildsBoxes) {
    const std::vector<BvhBox> boxes = test::Boxes(700, 8);
    ForEachDevice([&](Device& device) {
        const Ref<Buffer> buffer = Upload(device, boxes);
        const std::vector<Geometry> geometries = {
            device::AccelBoxes{.boxes = *buffer, .count = uint32_t(boxes.size())}};
        const LbvhBuilder builder(device, uint32_t(boxes.size()));
        const Bytes bvh = Build(device, builder, geometries).front();
        ExpectWellFormed(bvh, uint32_t(boxes.size()));
        EXPECT_EQ(test::Header(bvh).kind, kBvhBoxes);
    });
}

TEST(LbvhTest, BuildsEveryWorldsInstancesUpToItsCount) {
    const std::vector<uint32_t> capacities = {4, 0, 30, 7};
    const std::vector<uint32_t> counts = {4, 0, 21, 1};
    std::vector<test::Instances> worlds;
    std::vector<InstanceRecord> all;
    for (uint32_t w = 0; w < capacities.size(); ++w) {
        worlds.push_back(test::Scatter(capacities[w], w));
        all.insert(all.end(), worlds[w].records.begin(), worlds[w].records.end());
    }
    const uint32_t hidden = 4 + 3;
    all[hidden].custom &= kInstanceCustomMask;

    ForEachDevice([&](Device& device) {
        std::vector<BvhHeader> headers;
        for (const test::Instances& world : worlds) {
            for (const BvhBox& bounds : world.bounds) {
                BvhHeader header{};
                std::copy(std::begin(bounds.lower), std::end(bounds.lower), header.lower);
                std::copy(std::begin(bounds.upper), std::end(bounds.upper), header.upper);
                headers.push_back(header);
            }
        }
        const Ref<Buffer> blases = Upload(device, headers);
        std::vector<InstanceRecord> placed = all;
        for (size_t i = 0; i < placed.size(); ++i) {
            placed[i].blas = blases->Address() + i * sizeof(BvhHeader);
        }
        const Ref<Buffer> instances = Upload(device, placed);
        const Ref<Buffer> written = Upload(device, counts);
        const LbvhBuilder builder(device, uint32_t(all.size()), uint32_t(capacities.size()));
        for (const bool limited : {false, true}) {
            SCOPED_TRACE(limited ? "limited by counts" : "every instance");
            uint64_t bytes = 0;
            std::vector<uint64_t> starts;
            for (const uint32_t capacity : capacities) {
                starts.push_back(bytes);
                bytes += StorageBytes(capacity, kBvhInstances);
            }
            const Ref<Buffer> output = device.CreateBuffer({.bytes = bytes});
            std::vector<BufferRange> outputs;
            for (uint32_t w = 0; w < capacities.size(); ++w) {
                outputs.push_back(
                    output->Slice(starts[w], StorageBytes(capacities[w], kBvhInstances)));
            }
            CommandEncoder encoder = device.GetQueue().CreateEncoder();
            builder.Build(encoder, *instances, capacities,
                          limited ? std::optional<BufferRange>(*written) : std::nullopt, outputs);
            device.GetQueue().Submit(encoder.Finish());

            for (uint32_t w = 0; w < capacities.size(); ++w) {
                SCOPED_TRACE(w);
                const uint32_t live = limited ? counts[w] : capacities[w];
                std::vector<BvhBox> boxes = worlds[w].WorldBoxes();
                boxes.resize(live);
                const uint32_t masked = w == 2 && live > 3 ? 1 : 0;
                ExpectWellFormed(test::ReadBytes(device, outputs[w]), live - masked, boxes);
            }
        }
    });
}

TEST(LbvhTest, RejectsWhatItCannotBuild) {
    ForEachDevice([](Device& device) {
        EXPECT_THROW(LbvhBuilder(device, LbvhBuilder::kMaxPrimitives + 1), Error);
        EXPECT_THROW(LbvhBuilder(device, 10, 0), Error);

        const LbvhBuilder builder(device, 10, 2);
        Inputs inputs;
        inputs.Add(device, test::Soup(6, 1));
        inputs.Add(device, test::Soup(6, 2));
        const Ref<Buffer> boxes = Upload(device, test::Boxes(2, 3));
        const Ref<Buffer> output = device.CreateBuffer({.bytes = 4096});
        const std::vector<BufferRange> one = {*output};
        const std::vector<BufferRange> two = {output->Slice(0, 2048), output->Slice(2048)};
        const std::vector<BufferRange> three = {*output, *output, *output};
        const Geometry box = device::AccelBoxes{.boxes = *boxes, .count = 2};
        Geometry strided = inputs.geometries[0];
        std::get<device::AccelTriangles>(strided).stride = 10;
        const std::vector<Geometry> mixed = {inputs.geometries[0], box};
        const std::vector<Geometry> tooMany = {box, box, box};
        const std::vector<Geometry> tooLarge = {inputs.geometries[0], inputs.geometries[1]};

        CommandEncoder encoder = device.GetQueue().CreateEncoder();
        EXPECT_THROW(builder.Build(encoder, mixed, two), Error);
        EXPECT_THROW(builder.Build(encoder, tooMany, three), Error);
        EXPECT_THROW(builder.Build(encoder, std::span(inputs.geometries).first(1), two), Error);
        EXPECT_THROW(builder.Build(encoder, tooLarge, two), Error);
        EXPECT_THROW(builder.Build(encoder, std::span(&strided, 1), one), Error);
        EXPECT_THROW(builder.Build(encoder, std::span(inputs.geometries).first(1),
                                   std::vector<BufferRange>{output->Slice(0, 64)}),
                     Error);
        EXPECT_THROW(builder.Build(encoder, std::span<const Geometry>(), {}), Error);
        EXPECT_TRUE(encoder.Finish().Empty());
    });
}

}
}

// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/bvh/lbvh.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "ernie/bvh/common.h"
#include "ernie/bvh/kernels.h"
#include "ernie/device/device.h"
#include "ernie/device/kernels/shared.h"
#include "ernie/device/program.h"
#include "ernie/device/slang/dispatch.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"

namespace ernie::bvh {

using device::Buffer;
using device::BufferRange;
using device::BufferUsage;
using device::CommandEncoder;
using device::Device;
using device::Kernel;

namespace {

uint32_t KeyBits(uint32_t segments) {
    return kLbvhCodeBits + uint32_t(std::bit_width(2 * segments - 1));
}

uint32_t KeyWords(uint32_t segments) {
    return KeyBits(segments) <= 32 ? 1 : 2;
}

class Layout {
  public:
    explicit Layout(uint64_t alignment) : m_alignment(alignment) {}

    uint64_t Take(uint64_t bytes) {
        const uint64_t at = m_bytes;
        m_bytes += RoundUp(bytes, m_alignment);
        return at;
    }

    uint64_t Bytes() const {
        return m_bytes;
    }

  private:
    uint64_t m_alignment;
    uint64_t m_bytes = 0;
};

uint32_t CheckedCapacity(uint32_t capacity, uint32_t segments) {
    if (capacity > LbvhBuilder::kMaxPrimitives) {
        throw Error("LbvhBuilder of {} primitives; it builds at most {}", capacity,
                    LbvhBuilder::kMaxPrimitives);
    }
    if (segments == 0) {
        throw Error("LbvhBuilder of no segments; it builds at least one");
    }
    return capacity;
}

}

LbvhBuilder::LbvhBuilder(Device& device, uint32_t capacity, uint32_t segments)
    : m_capacity(CheckedCapacity(capacity, segments)), m_segments(segments),
      m_program(device.CreateProgram({.modules = {"ernie.bvh.kernels.lbvh"},
                                      .library = &kBvhKernels,
                                      .entries = {"Prepare", "Morton", "Hierarchy", "FitBoxes",
                                                  "Seed", "Advance", "Collapse", "Finish"},
                                      .label = "lbvh"})),
      m_prepare(&m_program->Entry("Prepare")), m_morton(&m_program->Entry("Morton")),
      m_hierarchy(&m_program->Entry("Hierarchy")), m_fitBoxes(&m_program->Entry("FitBoxes")),
      m_seed(&m_program->Entry("Seed")), m_advance(&m_program->Entry("Advance")),
      m_collapse(&m_program->Entry("Collapse")), m_finish(&m_program->Entry("Finish")),
      m_sort(device, capacity, 4 * KeyWords(segments)),
      m_state(device.CreateBuffer(
          {.bytes = sizeof(LbvhState), .usage = BufferUsage::Indirect, .label = "lbvh state"})) {
    m_params.keyWords = KeyWords(segments);

    const uint64_t n = std::max(capacity, 1u);
    Layout scratch(device::kBufferAlignment);
    const uint64_t boxes = scratch.Take(6 * n * 4);
    const uint64_t keys = scratch.Take(m_params.keyWords * n * 4);
    const uint64_t order = scratch.Take(n * 4);
    const uint64_t nodes = scratch.Take(kLbvhNodeWords * n * 4);
    const uint64_t parents = scratch.Take(n * 4);
    const uint64_t arrivals = scratch.Take(n * 4);
    const uint64_t tasks = scratch.Take(2 * kLbvhTaskWords * n * 4);
    const uint64_t bounds = scratch.Take(kLbvhBoundsWords * uint64_t(segments) * 4);
    const uint64_t roots = scratch.Take(uint64_t(segments) * 4);
    const uint64_t counters = scratch.Take(2 * uint64_t(segments) * 4);
    m_scratch = device.CreateBuffer({.bytes = scratch.Bytes(), .label = "lbvh scratch"});

    Layout layout(4);
    const uint64_t headers = layout.Take(uint64_t(segments) * sizeof(uint64_t));
    const uint64_t geometries = layout.Take(uint64_t(segments) * sizeof(LbvhGeometry));
    const uint64_t offsets = layout.Take((uint64_t(segments) + 1) * sizeof(uint32_t));
    m_layout = device.CreateBuffer({.bytes = layout.Bytes(), .label = "lbvh layout"});

    const uint64_t base = m_scratch->Address();
    m_params.boxes = {base + boxes};
    m_params.keys = {base + keys};
    m_params.order = {base + order};
    m_params.nodes = {base + nodes};
    m_params.parents = {base + parents};
    m_params.arrivals = {base + arrivals};
    m_params.tasks = {base + tasks};
    m_params.queue = uint32_t(n);
    m_params.bounds = {base + bounds};
    m_params.roots = {base + roots};
    m_params.counters = {base + counters};
    m_params.headers = {m_layout->Address() + headers};
    m_params.geometries = {m_layout->Address() + geometries};
    m_params.offsets = {m_layout->Address() + offsets};
    m_params.state = {m_state->Address()};
}

LbvhGeometry LbvhEntry(device::Program& program, const char* what, const Geometry& geometry) {
    LbvhGeometry entry{};
    if (const auto* set = std::get_if<device::AccelBoxes>(&geometry)) {
        entry.data =
            device::RequireRange<float>(program, what, set->boxes, uint64_t(set->count) * 6);
        entry.stride = 6;
        return entry;
    }
    const device::AccelTriangles& mesh = std::get<device::AccelTriangles>(geometry);
    if (mesh.stride < 12 || mesh.stride % 4 != 0) {
        throw Error("{} of a mesh of vertices {} bytes apart; they must be at least 12 and a "
                    "multiple of 4",
                    what, mesh.stride);
    }
    const uint64_t corners = mesh.indices ? mesh.vertexCount : uint64_t(mesh.count) * 3;
    if (corners > 0) {
        device::RequireRange<std::byte>(program, what, mesh.vertices,
                                        (corners - 1) * mesh.stride + 12);
    }
    entry.data = {mesh.vertices.Address()};
    entry.stride = mesh.stride / 4;
    if (mesh.indices) {
        entry.indices =
            device::RequireRange<uint32_t>(program, what, *mesh.indices, uint64_t(mesh.count) * 3);
    }
    return entry;
}

BufferRange LbvhBuilder::Words(uint64_t at, uint64_t words) const {
    return m_scratch->Slice(at - m_scratch->Address(), words * 4);
}

void LbvhBuilder::Build(CommandEncoder& encoder, std::span<const Geometry> geometries,
                        std::span<const BufferRange> outputs) const {
    LbvhParams params = m_params;
    params.kind = geometries.empty() ? kBvhTriangles : Kind(geometries[0]);
    std::vector<uint32_t> capacities;
    std::vector<LbvhGeometry> table;
    for (size_t i = 0; i < geometries.size(); ++i) {
        if (Kind(geometries[i]) != params.kind) {
            throw Error("Build of both triangles and boxes; a build takes one or the other");
        }
        table.push_back(LbvhEntry(*m_program, "Build", geometries[i]));
        capacities.push_back(PrimitiveCount(geometries[i]));
    }

    Record(encoder, params, capacities, outputs, table);
}

void LbvhBuilder::Build(CommandEncoder& encoder, BufferRange instances,
                        std::span<const uint32_t> capacities, std::optional<BufferRange> counts,
                        std::span<const BufferRange> outputs) const {
    const uint64_t total = Total(capacities);
    LbvhParams params = m_params;
    params.kind = kBvhInstances;
    params.instances = device::RequireRange<InstanceRecord>(*m_program, "Build", instances, total);
    if (counts) {
        params.counts =
            device::RequireRange<uint32_t>(*m_program, "Build", *counts, capacities.size());
    }
    Record(encoder, params, capacities, outputs, {});
}

void LbvhBuilder::Record(CommandEncoder& encoder, LbvhParams params,
                         std::span<const uint32_t> capacities, std::span<const BufferRange> outputs,
                         std::span<const LbvhGeometry> geometries) const {
    ERNIE_ZONE("Bvh/Lbvh");
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Bvh/Lbvh");
    const auto segments = uint32_t(capacities.size());
    if (segments == 0 || segments > m_segments) {
        throw Error("Build of {} segments; it was made for 1 to {}", segments, m_segments);
    }
    if (outputs.size() != segments) {
        throw Error("Build of {} segments into {} outputs", segments, outputs.size());
    }
    const uint64_t total = Total(capacities);
    if (total > m_capacity) {
        throw Error("Build of {} primitives; it was made for at most {}", total, m_capacity);
    }

    std::vector<uint64_t> headers;
    headers.reserve(segments);
    for (uint32_t s = 0; s < segments; ++s) {
        headers.push_back(device::RequireRange<std::byte>(*m_program, "Build", outputs[s],
                                                          StorageBytes(capacities[s], params.kind))
                              .address);
    }

    params.total = uint32_t(total);
    params.segments = segments;
    const device::ZoneScope scope(encoder, zone);
    const std::vector<uint32_t> offsets = Offsets(capacities);
    const auto update = [&](uint64_t address, const void* data, uint64_t bytes) {
        const uint64_t offset = address - m_layout->Address();
        for (uint64_t done = 0; done < bytes; done += CommandEncoder::kMaxUpdateBytes) {
            const uint64_t part = std::min(bytes - done, CommandEncoder::kMaxUpdateBytes);
            encoder.Update(m_layout->Slice(offset + done, part),
                           static_cast<const std::byte*>(data) + done);
        }
    };

    const uint32_t n = std::max(params.total, 1u);
    const LbvhState start{.parity = 1};
    encoder.Barrier();
    update(params.headers.address, headers.data(), headers.size() * 8);
    update(params.geometries.address, geometries.data(), geometries.size() * sizeof(LbvhGeometry));
    update(params.offsets.address, offsets.data(), offsets.size() * 4);
    encoder.Update(*m_state, &start);
    encoder.Fill(Words(params.bounds.address, uint64_t(kLbvhBoundsWords) * segments), 0xFFFFFFFF);
    encoder.Fill(Words(params.roots.address, segments), kLbvhNone);
    encoder.Fill(Words(params.counters.address, uint64_t(2) * segments), 0);
    encoder.Fill(Words(params.parents.address, n), kLbvhNone);
    encoder.Fill(Words(params.arrivals.address, n), 0);

    if (params.total > 0) {
        const device::Size3 each{.x = device::GroupCount(params.total, kLbvhThreads)};
        encoder.Barrier();
        encoder.Dispatch(*m_prepare, params, each);
        encoder.Barrier();
        encoder.Dispatch(*m_morton, params, each);
        m_sort.Sort(encoder, Words(params.keys.address, uint64_t(params.keyWords) * n),
                    Words(params.order.address, n), params.total, KeyBits(segments));
        encoder.Barrier();
        encoder.Dispatch(*m_hierarchy, params, each);
        encoder.Barrier();
        encoder.Dispatch(*m_fitBoxes, params, each);
        encoder.Barrier();
        encoder.Dispatch(*m_seed, params, {.x = device::GroupCount(segments, kLbvhThreads)});

        const device::Size3 most{
            .x = device::GroupCount(std::min(params.total, segments + params.total / 4),
                                    kLbvhThreads)};
        for (uint32_t level = 0; level < kLbvhLevels; ++level) {
            encoder.Barrier();
            encoder.Dispatch(*m_advance, params, {});
            encoder.Barrier();
            encoder.Dispatch(*m_collapse, params, most,
                             m_state->Slice(offsetof(LbvhState, groups), 12));
        }
    }

    encoder.Barrier();
    encoder.Dispatch(*m_finish, params, {.x = device::GroupCount(segments, kLbvhThreads)});
}

}

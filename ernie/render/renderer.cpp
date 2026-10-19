// SPDX-License-Identifier: BSD-3-Clause
/// @file renderer.cpp
/// @brief Rendering a scene's products: the device scene brought up to date, products alike in
/// size and vars traced together into stacked columns, and the columns read back on request.
#include "ernie/render/renderer.h"

#include "ernie/device/bindless.h"
#include "ernie/device/context.h"
#include "ernie/device/staging.h"
#include "ernie/render/device_scene.h"
#include "ernie/render/products.h"
#include "ernie/render/rt/camera_pass.h"
#include "ernie/scene/schema/render.h"
#include "ernie/util/profiler.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace ernie {
namespace {

/// How a var is stored, and the camera output that writes it.
struct VarFormat {
    std::string_view name;
    DType type;
    std::vector<std::int64_t> channels;
    std::optional<graph::Buffer> CameraOutputs::* output;
};

const std::array<VarFormat, 5>& VarFormats() {
    static const std::array<VarFormat, 5> formats{{
        {render_vars::kLdrColor, DType::UInt8, {4}, &CameraOutputs::ldr},
        {render_vars::kDistanceToCamera, DType::Float32, {}, &CameraOutputs::distances},
        {render_vars::kNormal, DType::Float32, {3}, &CameraOutputs::normals},
        // kNoInstance reads as -1.
        {render_vars::kPrimId, DType::Int32, {}, &CameraOutputs::prims},
        {render_vars::kAlbedo, DType::Float32, {CameraPass::kBands}, &CameraOutputs::albedos},
    }};
    return formats;
}

const VarFormat& FindFormat(std::string_view name) {
    return *std::ranges::find(VarFormats(), name, &VarFormat::name);
}

/// A batch and the views and columns it renders.
struct Batch {
    RenderBatch outputs;
    std::vector<CameraView> views;
    std::vector<std::unique_ptr<DeviceColumn>> columns;
};

/// @return The products grouped into batches of one size and vars, in order of first appearance.
std::vector<std::vector<const ProductSpec*>> GroupProducts(const std::vector<ProductSpec>& specs) {
    std::vector<std::vector<const ProductSpec*>> groups;
    for (const ProductSpec& spec : specs) {
        const auto alike = std::ranges::find_if(groups, [&](const auto& group) {
            return group.front()->size == spec.size && group.front()->vars == spec.vars;
        });
        if (alike != groups.end()) {
            alike->push_back(&spec);
        } else {
            groups.push_back({&spec});
        }
    }
    return groups;
}

}  // namespace

const RenderVarOutput* RenderBatch::Find(std::string_view name) const {
    const auto found = std::ranges::find(vars, name, &RenderVarOutput::name);
    return found != vars.end() ? &*found : nullptr;
}

/// What a renderer keeps on its device.
struct Renderer::Device {
    ShaderCompiler compiler;
    std::unique_ptr<DeviceContext> context;
    /// Where the device scene and outputs live: shared with CUDA where the device can export
    /// memory, so bound columns and outputs map in place.
    std::unique_ptr<Arenas> arenas;
    std::unique_ptr<BindlessHeap> heap;
    std::unique_ptr<CameraPass> camera;
    /// The scene `deviceScene` follows, by `Scene::Id`: an address can be a new scene's.
    std::uint64_t sceneId = 0;
    std::unique_ptr<DeviceScene> deviceScene;
    std::vector<Batch> batches;
    std::vector<RenderBatch> rendered;
    /// What the next step waits for.
    std::vector<Ticket> waits;

    /// @return The device scene following `scene`, made anew for a scene it does not follow.
    DeviceScene& Follow(const Scene& scene);
    /// Makes the batches for `specs`, keeping last step's columns where their vars match.
    void Plan(const Scene& scene, const std::vector<ProductSpec>& specs);
    /// Sizes a batch's columns for its views, making them when there are none.
    void Allocate(Batch& batch, const std::vector<std::string>& vars);
};

void Renderer::Device::Plan(const Scene& scene, const std::vector<ProductSpec>& specs) {
    const std::vector<std::vector<const ProductSpec*>> groups = GroupProducts(specs);
    std::vector<Batch> planned(groups.size());
    for (std::size_t i = 0; i < groups.size(); ++i) {
        Batch& batch = planned[i];
        batch.outputs.size = groups[i].front()->size;
        for (const ProductSpec* spec : groups[i]) {
            batch.outputs.products.push_back(spec->product);
            batch.views.push_back({.cameraToWorld = scene.WorldTransform(spec->camera),
                                   .size = spec->size,
                                   .lens = spec->lens,
                                   .node = spec->camera.slot,
                                   .world = spec->world});
        }
        if (i < batches.size()) {
            batch.columns = std::move(batches[i].columns);
        }
        Allocate(batch, groups[i].front()->vars);
    }
    batches = std::move(planned);
    rendered.clear();
    for (const Batch& batch : batches) {
        rendered.push_back(batch.outputs);
    }
}

void Renderer::Device::Allocate(Batch& batch, const std::vector<std::string>& vars) {
    const glm::uvec2 size = batch.outputs.size;
    const std::uint64_t pixels = batch.views.size() * std::uint64_t{size.x} * size.y;
    const auto views = static_cast<std::int64_t>(batch.views.size());
    batch.columns.resize(vars.size());
    for (std::size_t i = 0; i < vars.size(); ++i) {
        const VarFormat& format = FindFormat(vars[i]);
        std::unique_ptr<DeviceColumn>& column = batch.columns[i];
        if (!column || column->Name() != format.name) {
            column = std::make_unique<DeviceColumn>(*arenas, format.name, format.type,
                                                    format.channels, 0);
        }
        column->Resize(pixels, context->Main());
        std::vector<std::int64_t> shape{views, size.y, size.x};
        shape.insert(shape.end(), format.channels.begin(), format.channels.end());
        batch.outputs.vars.push_back({.name = std::string(format.name),
                                      .type = format.type,
                                      .shape = std::move(shape),
                                      .column = column.get()});
    }
}

Renderer::Renderer(const RendererOptions& options) : device_(std::make_unique<Device>()) {
    Device& device = *device_;
    device.context = std::make_unique<DeviceContext>(
        DeviceContextOptions{.validation = options.validation, .device = options.device});
    if (!device.context->Capabilities().rayQuery) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          device.context->Info().name + " cannot ray trace");
    }
    device.arenas = std::make_unique<Arenas>(
        *device.context,
        device.context->Capabilities().externalMemory ? MemoryUse::Shared : MemoryUse::Device,
        "renderer", std::uint64_t{16} << 20);
    device.heap = std::make_unique<BindlessHeap>(*device.context);
    device.camera = std::make_unique<CameraPass>(*device.context, device.compiler, *device.heap);
}

Renderer::~Renderer() {
    device_->context->WaitIdle();
}

DeviceContext& Renderer::Context() {
    return *device_->context;
}

DeviceScene& Renderer::Device::Follow(const Scene& scene) {
    if (scene.Id() != sceneId) {
        context->WaitIdle();
        deviceScene = std::make_unique<DeviceScene>(*context, *arenas, scene, compiler);
        sceneId = scene.Id();
        batches.clear();
        rendered.clear();
    }
    return *deviceScene;
}

std::span<const RenderBatch> Renderer::Step(Scene& scene) {
    ERNIE_ZONE("Render step");
    scene.Update();
    const std::vector<ProductSpec> specs = ReadProducts(scene);
    Device& device = *device_;
    DeviceScene& deviceScene = device.Follow(scene);
    device.Plan(scene, specs);
    const auto highest = std::ranges::max(specs, {}, &ProductSpec::world).world;
    deviceScene.Instances().ReserveWorlds(highest + 1);
    deviceScene.Follow(glm::dvec3(device.batches.front().views.front().cameraToWorld[3]));
    deviceScene.Execute(
        [&](graph::Graph& graph) {
            for (const Batch& batch : device.batches) {
                CameraOutputs outputs;
                for (std::size_t i = 0; i < batch.columns.size(); ++i) {
                    outputs.*FindFormat(batch.outputs.vars[i].name).output =
                        graph.Import(*batch.columns[i]);
                }
                device.camera->Add(graph, batch.views, outputs);
            }
        },
        device.waits);
    device.waits.clear();
    return device.rendered;
}

const DeviceColumn& Renderer::BindTransformOp(Scene& scene, const Path& prim,
                                              std::string_view attribute) {
    DeviceScene& deviceScene = device_->Follow(scene);
    const BucketId bucket = scene.Store().PrimBucket(scene.Require(prim));
    const DeviceColumn& column = deviceScene.Columns().Bind(bucket, attribute);
    deviceScene.Transforms().Bind(bucket, attribute, column);
    return column;
}

void Renderer::WaitFor(const Ticket& ticket) {
    device_->waits.push_back(ticket);
}

Tensor Renderer::Read(const RenderVarOutput& var) {
    DeviceContext& context = *device_->context;
    const Ticket done = context.Main().Last();
    Readback readback(context, context.Main(), var.column->Storage(), var.column->Range().offset,
                      var.column->Bytes(), std::span(&done, 1));
    Tensor tensor = Tensor::Zeros(var.type, var.shape);
    std::ranges::copy(readback.Bytes().first(var.column->Bytes()),
                      static_cast<std::byte*>(tensor.Data()));
    return tensor;
}

Tensor Renderer::Render(Scene& scene) {
    const RenderBatch& first = Step(scene).front();
    const RenderVarOutput* ldr = first.Find(render_vars::kLdrColor);
    if (ldr == nullptr) {
        throw std::invalid_argument("'" + first.products.front().String() + "' holds no LdrColor");
    }
    return Read(*ldr).Row(0);
}

}  // namespace ernie

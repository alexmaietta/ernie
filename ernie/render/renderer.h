// SPDX-License-Identifier: BSD-3-Clause
/// @file renderer.h
/// @brief Rendering a scene's render products: to a host image in one call, or a step at a time
/// into device columns CUDA and PyTorch read in place.
///
///     Renderer renderer;
///     Tensor image = renderer.Render(scene);  // height x width x 4 bytes of LdrColor
///     for (const RenderBatch& batch : renderer.Step(scene)) {
///         Tensor depth = renderer.Read(*batch.Find(render_vars::kDistanceToCamera));
///     }
#pragma once

#include "ernie/export.h"
#include "ernie/scene/scene.h"
#include "ernie/util/tensor.h"

#include <glm/vec2.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

class DeviceColumn;
class DeviceContext;
struct Ticket;

/// @brief The most a renderer's scene holds, declared up front as batched simulation does: memory
/// is laid out once, never again as counts move, and a scene beyond it is an error, never cut
/// short. 0 for as many as there are.
struct RenderCapacities {
    std::uint32_t worlds = 0;
    /// Each world's instances, its own and the shared ones.
    std::uint32_t instancesPerWorld = 0;
    /// The products a step renders, a view each.
    std::uint32_t views = 0;
};

/// @brief How batch worlds trace the prims in every world, such as a shared ground and room. A
/// scene of one world keeps them in its top level either way.
enum class SharedPrims : std::uint8_t {
    /// In every world's top level, beside its own prims.
    Duplicated,
    /// In one top level of their own, traced after each world's: a world's top level holds only
    /// its own prims, however many shared ones there are. Measured faster from 64 to 16,384
    /// worlds, by up to 40% with 64 shared prims (`render.shared_prims` in ernie_benchmarks).
    SharedTopLevel,
};

struct RendererOptions {
    /// The device: its UUID in hex, or part of its name; empty for the best scoring one.
    std::string device;
    /// The Khronos validation layer with synchronization validation, when installed.
    bool validation = false;
    RenderCapacities capacities;
    /// The sets of outputs steps take turns rendering into: with 2, a learner reads one step's
    /// outputs while the next step renders into the other set.
    std::uint32_t outputSets = 1;
    /// How worlds trace their shared prims; the default is the faster one as measured.
    SharedPrims sharedPrims = SharedPrims::SharedTopLevel;
};

/// @brief One var of a batch: every view's pixels, stacked on the device.
struct RenderVarOutput {
    /// Its source name, one of `render_vars`.
    std::string name;
    DType type = DType::UInt8;
    /// (views, rows, cols), then the pixel's channels when it has several.
    std::vector<std::int64_t> shape;
    /// Its rows, a pixel each, in shared memory where the device can export it.
    const DeviceColumn* column = nullptr;
};

/// @brief Products of one size and vars, rendered in one dispatch: its views, one a product, in
/// the settings' order.
struct RenderBatch {
    std::vector<Path> products;
    glm::uvec2 size{1};
    std::vector<RenderVarOutput> vars;

    /// @return The var with this source name, or null.
    const RenderVarOutput* Find(std::string_view name) const;
};

/// @brief A device and what it keeps of the scene it last rendered, so rendering the same scene
/// again costs only what changed in it.
class ERNIE_API Renderer {
  public:
    /// @throws std::runtime_error without a device that can ray trace.
    explicit Renderer(const RendererOptions& options = {});
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    /// @brief Renders the products of the scene's first `RenderSettings` (`ReadProducts`) as
    /// Minimal sees them: albedo lit from the eye. It updates the scene first. Products alike in
    /// size and vars render in one dispatch, and the step in one submission; it returns once that
    /// is submitted. Cameras take their poses from the device's transforms, so poses CUDA writes
    /// move them. Steps take turns among the output sets.
    /// @return The batches, valid until a later step renders into their set again: the next step
    /// with one set, the one after with two.
    /// @throws std::invalid_argument as `ReadProducts`; std::length_error for a scene beyond the
    /// declared capacities.
    std::span<const RenderBatch> Step(Scene& scene);

    /// @return A var's pixels copied to the host once the last step has rendered them.
    Tensor Read(const RenderVarOutput& var);

    /// @brief Hands an attribute to the device for every prim in `prim`'s bucket (the prims of its
    /// type and applied schemas), as the device scene's columns bind it: from then on the column
    /// is what steps read, so CUDA, PyTorch or Warp write it in place each step, as poses or
    /// domain randomisation do, and nothing goes through the host. Values the host authors later
    /// still overwrite their rows. Binds a transform op in the prims' `xformOpOrder`, or
    /// `primvars:displayColor`.
    /// @return The column, a row a prim in the bucket's row order, in memory CUDA can map.
    /// @throws std::invalid_argument for another attribute, or as `SceneColumns::Bind` and
    /// `TransformGraph::Bind`.
    const DeviceColumn& BindColumn(Scene& scene, const Path& prim, std::string_view attribute);
    /// @brief Makes the next step wait on the device for `ticket`, such as CUDA's writes to a
    /// bound column.
    void WaitFor(const Ticket& ticket);
    /// @return Whether the last step submitted the commands recorded for an earlier one again,
    /// as steps do while nothing changes on the host: its cost then is one submission.
    bool Replayed() const;

    /// @brief Steps and reads the first product's LdrColor.
    /// @return LdrColor, height x width x 4 bytes (RGBA8, display encoded) on the host.
    /// @throws std::invalid_argument as `Step`, or when that product holds no LdrColor.
    Tensor Render(Scene& scene);

    /// @return Its device, for Ernie's own tests and tools.
    DeviceContext& Context();

  private:
    struct Device;
    std::unique_ptr<Device> device_;
};

}  // namespace ernie

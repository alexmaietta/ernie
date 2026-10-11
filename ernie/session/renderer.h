// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/session/delegate.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// Options for creating a `Renderer`.
struct RendererOptions {
    std::string backend =
        "raster";  ///< Render backend, one of `RenderBackends()`.
    std::string device;       ///< GPU name to prefer; empty picks a
                              ///< discrete GPU first.
    bool validation = false;  ///< Enable the Vulkan validation layer
                              ///< when installed.
};

/// Ernie's front door for rendering a `Scene` headless: a GPU and a
/// render backend on it, with no graphics API to set up. Step the
/// scene's UsdRender products (one image each, or a batch of views,
/// `ViewPosesOf`) and map their render vars to the CPU or CUDA:
///
///     ernie::Renderer renderer;
///     ernie::RenderProductSetOutputs outputs = renderer.Step(scene);
///     ernie::Tensor image = outputs.at("/Render/Product")
///                               .vars.at("LdrColor")
///                               .Map(ernie::MapDevice::Cpu).tensor;
///
/// Applications that draw into their own windows or command buffers
/// use the `RenderDelegate` and its `VulkanContext` directly
/// (`Delegate`, `Context`), as the viewer does.
class Renderer {
  public:
    /// @throws std::runtime_error without a usable GPU, or
    /// std::invalid_argument for an unknown backend.
    explicit Renderer(const RendererOptions& options = {});
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) noexcept;
    Renderer& operator=(Renderer&&) noexcept;

    /// The backend's name and abilities.
    const RendererInfo& Info() const;

    /// The GPU's name.
    std::string DeviceName() const;

    /// Renders `products` of `scene` (all the settings' products when
    /// empty) and waits for them; see `RenderDelegate::StepAsync`.
    RenderProductSetOutputs Step(const Scene& scene,
                                 std::span<const Path> products = {},
                                 const Path& settings = {});

    /// Renders `products` of `scene` and returns at once; see
    /// `RenderDelegate::StepAsync`.
    StepOperation StepAsync(const Scene& scene,
                            std::span<const Path> products = {},
                            const Path& settings = {});

    /// Renders the settings' first product and returns its LdrColor;
    /// see `RenderDelegate::Render`.
    Tensor Render(const Scene& scene, const Path& settings = {},
                  Device device = {});

    /// Uploads what changed in `scene` now, to check new data before
    /// stepping.
    void Sync(const Scene& scene);

    /// The prim path of a PrimId render var value.
    Path PrimPath(std::uint32_t id) const;

    RenderStats Stats() const;

    /// Validation messages since the last call, with
    /// `RendererOptions::validation`.
    std::vector<std::string> TakeValidationMessages();

    /// The backend itself, for interactive drawing and what the
    /// renderer does not wrap.
    RenderDelegate& Delegate();
    const RenderDelegate& Delegate() const;

    /// The GPU device the backend renders on.
    VulkanContext& Context();
    const VulkanContext& Context() const;

  private:
    std::unique_ptr<VulkanContext> context_;
    std::unique_ptr<RenderDelegate> delegate_;
};

}  // namespace ernie

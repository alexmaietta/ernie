// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/renderer.h"

#include "ernie/device/vulkan_context.h"

#include <utility>

namespace ernie {

namespace {

std::unique_ptr<VulkanContext>
CreateContext(const RendererOptions& options) {
    VulkanContextOptions context;
    context.validation = options.validation;
    context.preferredDevice = options.device;
    return VulkanContext::Create(context);
}

}  // namespace

Renderer::Renderer(const RendererOptions& options)
    : context_(CreateContext(options)),
      delegate_(CreateRenderDelegate(options.backend, *context_)) {}

// The delegate goes before the context it renders with.
Renderer::~Renderer() {
    delegate_.reset();
}

Renderer::Renderer(Renderer&&) noexcept = default;

Renderer& Renderer::operator=(Renderer&& other) noexcept {
    if (this != &other) {
        delegate_.reset();
        context_ = std::move(other.context_);
        delegate_ = std::move(other.delegate_);
    }
    return *this;
}

const RendererInfo& Renderer::Info() const {
    return delegate_->Info();
}

std::string Renderer::DeviceName() const {
    return context_->DeviceName();
}

RenderProductSetOutputs Renderer::Step(const Scene& scene,
                                       std::span<const Path> products,
                                       const Path& settings) {
    return delegate_->Step(scene, products, settings);
}

StepOperation Renderer::StepAsync(const Scene& scene,
                                  std::span<const Path> products,
                                  const Path& settings) {
    return delegate_->StepAsync(scene, products, settings);
}

Tensor Renderer::Render(const Scene& scene, const Path& settings,
                        Device device) {
    return delegate_->Render(scene, settings, device);
}

void Renderer::Sync(const Scene& scene) {
    delegate_->Sync(scene);
}

Path Renderer::PrimPath(std::uint32_t id) const {
    return delegate_->PrimPath(id);
}

RenderStats Renderer::Stats() const {
    return delegate_->Stats();
}

std::vector<std::string> Renderer::TakeValidationMessages() {
    return context_->TakeValidationMessages();
}

RenderDelegate& Renderer::Delegate() {
    return *delegate_;
}

const RenderDelegate& Renderer::Delegate() const {
    return *delegate_;
}

VulkanContext& Renderer::Context() {
    return *context_;
}

const VulkanContext& Renderer::Context() const {
    return *context_;
}

}  // namespace ernie

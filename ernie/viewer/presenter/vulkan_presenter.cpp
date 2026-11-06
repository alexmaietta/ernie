// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <optional>

#include <volk.h>

#include <GLFW/glfw3.h>
#include <imgui_impl_vulkan.h>

#include "ernie/device/vulkan/native.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"
#include "ernie/viewer/presenter/presenter.h"

extern "C" {
extern const unsigned char ernie_viewport_spirv[];
extern const std::size_t ernie_viewport_spirv_size;
}

namespace ernie::app {
namespace {

namespace vulkan = device::vulkan;

constexpr uint32_t kImGuiImages = 3;

struct ViewportPush {
    uint64_t pixels;
    uint32_t size[2];
    float origin[2];
    float extent[2];
    float opacity;
    uint32_t blended;
};

struct ViewportDraw {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    uint64_t pixels;
    uint32_t width;
    uint32_t height;
    ImVec2 min;
    ImVec2 max;
    float opacity;
    uint32_t blended;
};

void CheckVk(VkResult result, const char* call) {
    if (result != VK_SUCCESS) {
        throw Error("{} failed with VkResult {}", call, int(result));
    }
}

template <class T, class Query> std::vector<T> Enumerate(Query query) {
    uint32_t count = 0;
    query(&count, nullptr);
    std::vector<T> items(count);
    query(&count, items.data());
    items.resize(count);
    return items;
}

VkSemaphore CreateBinarySemaphore(VkDevice device) {
    VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore semaphore = VK_NULL_HANDLE;
    CheckVk(vkCreateSemaphore(device, &info, nullptr, &semaphore), "vkCreateSemaphore");
    return semaphore;
}

VkImageView CreateView(VkDevice device, VkImage image, VkFormat format) {
    VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    CheckVk(vkCreateImageView(device, &info, nullptr, &view), "vkCreateImageView");
    return view;
}

VkCommandPool CreatePool(VkDevice device, uint32_t family) {
    VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    info.queueFamilyIndex = family;
    VkCommandPool pool = VK_NULL_HANDLE;
    CheckVk(vkCreateCommandPool(device, &info, nullptr, &pool), "vkCreateCommandPool");
    return pool;
}

VkCommandBuffer AllocateCommands(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    CheckVk(vkAllocateCommandBuffers(device, &allocation, &commands), "vkAllocateCommandBuffers");
    return commands;
}

void BeginCommands(VkCommandBuffer commands) {
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CheckVk(vkBeginCommandBuffer(commands, &begin), "vkBeginCommandBuffer");
}

uint32_t MemoryType(VkPhysicalDevice physical, uint32_t types, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    for (uint32_t type = 0; type < memory.memoryTypeCount; ++type) {
        if ((types & (1u << type)) != 0 &&
            (memory.memoryTypes[type].propertyFlags & flags) == flags) {
            return type;
        }
    }
    throw Error("No Vulkan memory type fits");
}

void TransitionImage(VkCommandBuffer commands, VkImage image, VkImageLayout from, VkImageLayout to,
                     VkPipelineStageFlags2 after, VkAccessFlags2 written,
                     VkPipelineStageFlags2 before, VkAccessFlags2 used) {
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = after;
    barrier.srcAccessMask = written;
    barrier.dstStageMask = before;
    barrier.dstAccessMask = used;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commands, &dependency);
}

void DrawViewport(const ImDrawList*, const ImDrawCmd* command) {
    const ViewportDraw& draw = *static_cast<const ViewportDraw*>(command->UserCallbackData);
    const auto* state = static_cast<const ImGui_ImplVulkan_RenderState*>(
        ImGui::GetPlatformIO().Renderer_RenderState);
    const ImDrawData& data = *ImGui::GetDrawData();
    const ImVec2 scale = data.FramebufferScale;
    const ImVec2 origin = data.DisplayPos;
    const float width = data.DisplaySize.x * scale.x;
    const float height = data.DisplaySize.y * scale.y;
    const float x0 = (draw.min.x - origin.x) * scale.x;
    const float y0 = (draw.min.y - origin.y) * scale.y;
    const float x1 = (draw.max.x - origin.x) * scale.x;
    const float y1 = (draw.max.y - origin.y) * scale.y;
    const float clipX0 = std::clamp((command->ClipRect.x - origin.x) * scale.x, 0.0f, width);
    const float clipY0 = std::clamp((command->ClipRect.y - origin.y) * scale.y, 0.0f, height);
    const float clipX1 = std::clamp((command->ClipRect.z - origin.x) * scale.x, 0.0f, width);
    const float clipY1 = std::clamp((command->ClipRect.w - origin.y) * scale.y, 0.0f, height);
    if (x1 <= x0 || y1 <= y0 || clipX1 <= clipX0 || clipY1 <= clipY0) {
        return;
    }

    const VkCommandBuffer commands = state->CommandBuffer;
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.pipeline);
    const VkViewport viewport{x0, y0, x1 - x0, y1 - y0, 0.0f, 1.0f};
    vkCmdSetViewport(commands, 0, 1, &viewport);
    const VkRect2D scissor{{int32_t(clipX0), int32_t(clipY0)},
                           {uint32_t(clipX1 - clipX0), uint32_t(clipY1 - clipY0)}};
    vkCmdSetScissor(commands, 0, 1, &scissor);
    const ViewportPush push{draw.pixels,  {draw.width, draw.height},
                            {x0, y0},     {x1 - x0, y1 - y0},
                            draw.opacity, draw.blended};
    vkCmdPushConstants(commands, draw.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
    vkCmdDraw(commands, 3, 1, 0, 0);
}

class VulkanPresenter : public Presenter {
  public:
    VulkanPresenter(device::Device& device, GLFWwindow* window, Extent size, bool vsync);
    ~VulkanPresenter() override;
    VulkanPresenter(const VulkanPresenter&) = delete;
    VulkanPresenter& operator=(const VulkanPresenter&) = delete;

    bool Begin() override;

    Extent Size() const override {
        return {m_extent.width, m_extent.height};
    }

    void Viewport(ImDrawList& list, const device::Tensor& pixels, ImVec2 min, ImVec2 max) override;
    void Image(ImDrawList& list, const device::Tensor& pixels, ImVec2 min, ImVec2 max,
               float opacity) override;
    device::Operation End(Span<const device::Operation> after) override;
    std::vector<uint8_t> ReadPixels(const device::Operation& after) override;

  private:
    struct SwapImage {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkSemaphore presentable = VK_NULL_HANDLE;
    };

    struct Slot {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer commands = VK_NULL_HANDLE;
        VkSemaphore acquired = VK_NULL_HANDLE;
        device::Operation done;
    };

    void Release() noexcept;
    void HoldQueues(const std::function<void()>& use);
    void CreateSurface();
    void CreateOffscreen(Extent size);
    void CreateSwapchain();
    void DestroyImages() noexcept;
    void CreatePipeline();
    void StartImGui();
    bool Acquire(VkSemaphore acquired);

    device::Device& m_device;
    vulkan::Native m_native;
    GLFWwindow* m_window;
    bool m_vsync;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_format = VK_FORMAT_R8G8B8A8_UNORM;
    VkColorSpaceKHR m_colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D m_extent{};
    std::vector<SwapImage> m_images;
    VkDeviceMemory m_offscreenMemory = VK_NULL_HANDLE;
    bool m_stale = false;
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipeline m_blendPipeline = VK_NULL_HANDLE;
    bool m_imgui = false;
    std::array<Slot, 2> m_slots{};
    uint32_t m_next = 0;
    uint32_t m_index = 0;
};

VulkanPresenter::VulkanPresenter(device::Device& device, GLFWwindow* window, Extent size,
                                 bool vsync)
    : m_device(device), m_native(vulkan::NativeHandles(device)), m_window(window), m_vsync(vsync) {
    if (!device.Presents()) {
        throw Error("{} wasn't created to present, so it shows nothing", device.Adapter().alias);
    }
    volkInitializeCustom(m_native.getInstanceProcAddr);
    volkLoadInstanceOnly(m_native.instance);
    volkLoadDevice(m_native.device);
    try {
        if (m_window != nullptr) {
            CreateSurface();
            CreateSwapchain();
        } else {
            CreateOffscreen(size);
        }
        CreatePipeline();
        for (Slot& slot : m_slots) {
            slot.pool = CreatePool(m_native.device, m_native.queueFamily);
            slot.commands = AllocateCommands(m_native.device, slot.pool);
            slot.acquired = CreateBinarySemaphore(m_native.device);
        }
        StartImGui();
    } catch (...) {
        Release();
        throw;
    }
}

VulkanPresenter::~VulkanPresenter() {
    Release();
}

void VulkanPresenter::Release() noexcept {
    HoldQueues([&] {
        vkDeviceWaitIdle(m_native.device);
        if (m_imgui) {
            ImGui_ImplVulkan_Shutdown();
            m_imgui = false;
        }
    });
    for (Slot& slot : m_slots) {
        vkDestroySemaphore(m_native.device, slot.acquired, nullptr);
        vkDestroyCommandPool(m_native.device, slot.pool, nullptr);
        slot = {};
    }
    vkDestroyPipeline(m_native.device, m_pipeline, nullptr);
    vkDestroyPipeline(m_native.device, m_blendPipeline, nullptr);
    vkDestroyPipelineLayout(m_native.device, m_layout, nullptr);
    m_pipeline = VK_NULL_HANDLE;
    m_blendPipeline = VK_NULL_HANDLE;
    m_layout = VK_NULL_HANDLE;
    DestroyImages();
    if (m_offscreenMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_native.device, m_offscreenMemory, nullptr);
        m_offscreenMemory = VK_NULL_HANDLE;
    }
    vkDestroySwapchainKHR(m_native.device, m_swapchain, nullptr);
    m_swapchain = VK_NULL_HANDLE;
    if (m_surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_native.instance, m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }
}

void VulkanPresenter::HoldQueues(const std::function<void()>& use) {
    if (m_native.displaySharesRender) {
        vulkan::Exclusive(m_device, [&](VkQueue) { use(); });
        return;
    }
    vulkan::Exclusive(m_device, [&](VkQueue) {
        vulkan::Exclusive(m_device, [&](VkQueue) { use(); }, vulkan::QueueRole::Render);
    });
}

void VulkanPresenter::CreateSurface() {
    CheckVk(glfwCreateWindowSurface(m_native.instance, m_window, nullptr, &m_surface),
            "glfwCreateWindowSurface");
    VkBool32 supported = VK_FALSE;
    CheckVk(vkGetPhysicalDeviceSurfaceSupportKHR(m_native.physicalDevice, m_native.queueFamily,
                                                 m_surface, &supported),
            "vkGetPhysicalDeviceSurfaceSupportKHR");
    if (supported == VK_FALSE) {
        throw Error("{} can't present to this window", m_device.Adapter().alias);
    }
    const std::vector<VkSurfaceFormatKHR> formats =
        Enumerate<VkSurfaceFormatKHR>([&](uint32_t* count, VkSurfaceFormatKHR* items) {
            vkGetPhysicalDeviceSurfaceFormatsKHR(m_native.physicalDevice, m_surface, count, items);
        });
    if (formats.empty()) {
        throw Error("The window's surface has no formats");
    }
    const auto chosen = std::ranges::find_if(formats, [](const VkSurfaceFormatKHR& format) {
        return (format.format == VK_FORMAT_B8G8R8A8_UNORM ||
                format.format == VK_FORMAT_R8G8B8A8_UNORM) &&
               format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    });
    const VkSurfaceFormatKHR format = chosen != formats.end() ? *chosen : formats.front();
    m_format = format.format;
    m_colorSpace = format.colorSpace;
}

void VulkanPresenter::CreateOffscreen(Extent size) {
    m_extent = {size.width, size.height};
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = m_format;
    info.extent = {m_extent.width, m_extent.height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    SwapImage& image = m_images.emplace_back();
    CheckVk(vkCreateImage(m_native.device, &info, nullptr, &image.image), "vkCreateImage");
    VkMemoryRequirements needs;
    vkGetImageMemoryRequirements(m_native.device, image.image, &needs);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = needs.size;
    allocation.memoryTypeIndex = MemoryType(m_native.physicalDevice, needs.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CheckVk(vkAllocateMemory(m_native.device, &allocation, nullptr, &m_offscreenMemory),
            "vkAllocateMemory");
    CheckVk(vkBindImageMemory(m_native.device, image.image, m_offscreenMemory, 0),
            "vkBindImageMemory");
    image.view = CreateView(m_native.device, image.image, m_format);
}

void VulkanPresenter::CreateSwapchain() {
    VkSurfaceCapabilitiesKHR capabilities;
    CheckVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_native.physicalDevice, m_surface,
                                                      &capabilities),
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(m_window, &width, &height);
    m_extent = capabilities.currentExtent;
    if (m_extent.width == UINT32_MAX) {
        m_extent.width = std::clamp(uint32_t(std::max(width, 0)), capabilities.minImageExtent.width,
                                    capabilities.maxImageExtent.width);
        m_extent.height =
            std::clamp(uint32_t(std::max(height, 0)), capabilities.minImageExtent.height,
                       capabilities.maxImageExtent.height);
    }
    if (m_extent.width == 0 || m_extent.height == 0) {
        return;
    }
    uint32_t count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        count = std::min(count, capabilities.maxImageCount);
    }
    VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if ((capabilities.supportedCompositeAlpha & alpha) == 0) {
        const uint32_t supported = capabilities.supportedCompositeAlpha;
        alpha = VkCompositeAlphaFlagBitsKHR(supported & (~supported + 1));
    }

    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = m_surface;
    info.minImageCount = count;
    info.imageFormat = m_format;
    info.imageColorSpace = m_colorSpace;
    info.imageExtent = m_extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = capabilities.currentTransform;
    info.compositeAlpha = alpha;
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (!m_vsync) {
        const std::vector<VkPresentModeKHR> modes =
            Enumerate<VkPresentModeKHR>([&](uint32_t* found, VkPresentModeKHR* items) {
                vkGetPhysicalDeviceSurfacePresentModesKHR(m_native.physicalDevice, m_surface, found,
                                                          items);
            });
        for (const VkPresentModeKHR mode :
             {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR}) {
            if (std::ranges::find(modes, mode) != modes.end()) {
                info.presentMode = mode;
                break;
            }
        }
    }
    info.clipped = VK_TRUE;
    info.oldSwapchain = m_swapchain;
    VkSwapchainKHR created = VK_NULL_HANDLE;
    CheckVk(vkCreateSwapchainKHR(m_native.device, &info, nullptr, &created),
            "vkCreateSwapchainKHR");
    DestroyImages();
    vkDestroySwapchainKHR(m_native.device, m_swapchain, nullptr);
    m_swapchain = created;
    const std::vector<VkImage> images = Enumerate<VkImage>([&](uint32_t* found, VkImage* items) {
        vkGetSwapchainImagesKHR(m_native.device, m_swapchain, found, items);
    });
    for (const VkImage image : images) {
        m_images.push_back({image, CreateView(m_native.device, image, m_format),
                            CreateBinarySemaphore(m_native.device)});
    }
    m_stale = false;
}

void VulkanPresenter::DestroyImages() noexcept {
    for (const SwapImage& image : m_images) {
        vkDestroyImageView(m_native.device, image.view, nullptr);
        vkDestroySemaphore(m_native.device, image.presentable, nullptr);
        if (m_window == nullptr) {
            vkDestroyImage(m_native.device, image.image, nullptr);
        }
    }
    m_images.clear();
}

void VulkanPresenter::CreatePipeline() {
    std::vector<uint32_t> code((ernie_viewport_spirv_size + 3) / 4);
    std::memcpy(code.data(), ernie_viewport_spirv, ernie_viewport_spirv_size);
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = ernie_viewport_spirv_size;
    moduleInfo.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    CheckVk(vkCreateShaderModule(m_native.device, &moduleInfo, nullptr, &module),
            "vkCreateShaderModule");

    const VkPushConstantRange range{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ViewportPush)};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    CheckVk(vkCreatePipelineLayout(m_native.device, &layoutInfo, nullptr, &m_layout),
            "vkCreatePipelineLayout");

    VkPipelineShaderStageCreateInfo stages[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = module;
    stages[0].pName = "VertexMain";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = module;
    stages[1].pName = "FragmentMain";
    VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewportState{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &attachment;
    const VkDynamicState dynamics[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamics;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &m_format;

    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewportState;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = m_layout;
    VkResult created =
        vkCreateGraphicsPipelines(m_native.device, VK_NULL_HANDLE, 1, &info, nullptr, &m_pipeline);
    if (created == VK_SUCCESS) {
        attachment.blendEnable = VK_TRUE;
        attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        attachment.colorBlendOp = VK_BLEND_OP_ADD;
        attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        attachment.alphaBlendOp = VK_BLEND_OP_ADD;
        created = vkCreateGraphicsPipelines(m_native.device, VK_NULL_HANDLE, 1, &info, nullptr,
                                            &m_blendPipeline);
    }
    vkDestroyShaderModule(m_native.device, module, nullptr);
    CheckVk(created, "vkCreateGraphicsPipelines");
}

void VulkanPresenter::StartImGui() {
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = m_native.apiVersion;
    info.Instance = m_native.instance;
    info.PhysicalDevice = m_native.physicalDevice;
    info.Device = m_native.device;
    info.QueueFamily = m_native.queueFamily;
    info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLER_POOL_SIZE;
    info.MinImageCount = 2;
    info.ImageCount = kImGuiImages;
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &m_format;
    info.CheckVkResultFn = [](VkResult result) { CheckVk(result, "ImGui's Vulkan backend"); };
    vulkan::Exclusive(m_device, [&](VkQueue queue) {
        info.Queue = queue;
        ImGui_ImplVulkan_Init(&info);
    });
    m_imgui = true;
}

bool VulkanPresenter::Acquire(VkSemaphore acquired) {
    if (m_window == nullptr) {
        m_index = 0;
        return true;
    }
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(m_window, &width, &height);
    if (width <= 0 || height <= 0) {
        glfwWaitEventsTimeout(0.05);
        return false;
    }
    if (m_stale || m_swapchain == VK_NULL_HANDLE || uint32_t(width) != m_extent.width ||
        uint32_t(height) != m_extent.height) {
        HoldQueues([&] { vkDeviceWaitIdle(m_native.device); });
        CreateSwapchain();
        if (m_swapchain == VK_NULL_HANDLE) {
            return false;
        }
    }
    const VkResult result = vkAcquireNextImageKHR(m_native.device, m_swapchain, UINT64_MAX,
                                                  acquired, VK_NULL_HANDLE, &m_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        m_stale = true;
        return false;
    }
    if (result == VK_SUBOPTIMAL_KHR) {
        m_stale = true;
    } else {
        CheckVk(result, "vkAcquireNextImageKHR");
    }
    return true;
}

bool VulkanPresenter::Begin() {
    Slot& slot = m_slots[m_next];
    slot.done.Wait();
    if (!Acquire(slot.acquired)) {
        return false;
    }
    ImGui_ImplVulkan_NewFrame();
    return true;
}

void VulkanPresenter::Viewport(ImDrawList& list, const device::Tensor& pixels, ImVec2 min,
                               ImVec2 max) {
    const Span<const int64_t> shape = pixels.Shape();
    const ViewportDraw draw{
        m_pipeline, m_layout, pixels.Address(), uint32_t(shape[1]), uint32_t(shape[0]), min, max,
        1.0f,       0};
    list.AddCallback(DrawViewport, const_cast<ViewportDraw*>(&draw), sizeof(draw));
    list.AddCallback(ImGui::GetPlatformIO().DrawCallback_ResetRenderState, nullptr);
}

void VulkanPresenter::Image(ImDrawList& list, const device::Tensor& pixels, ImVec2 min, ImVec2 max,
                            float opacity) {
    const Span<const int64_t> shape = pixels.Shape();
    const ViewportDraw draw{m_blendPipeline,
                            m_layout,
                            pixels.Address(),
                            uint32_t(shape[1]),
                            uint32_t(shape[0]),
                            min,
                            max,
                            opacity,
                            1};
    list.AddCallback(DrawViewport, const_cast<ViewportDraw*>(&draw), sizeof(draw));
    list.AddCallback(ImGui::GetPlatformIO().DrawCallback_ResetRenderState, nullptr);
}

device::Operation VulkanPresenter::End(Span<const device::Operation> after) {
    ERNIE_ZONE("Viewer/Present");
    Slot& slot = m_slots[m_next];
    m_next = (m_next + 1) % uint32_t(m_slots.size());
    const SwapImage& image = m_images[m_index];
    vulkan::Exclusive(m_device, [&](VkQueue) {
        const VkCommandBuffer commands = slot.commands;
        CheckVk(vkResetCommandPool(m_native.device, slot.pool, 0), "vkResetCommandPool");
        BeginCommands(commands);
        TransitionImage(commands, image.image, VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_NONE,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = image.view;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue.color = {{0.08f, 0.08f, 0.09f, 1.0f}};
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = {{0, 0}, m_extent};
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color;
        vkCmdBeginRendering(commands, &rendering);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commands);
        vkCmdEndRendering(commands);
        TransitionImage(commands, image.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        m_window != nullptr ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                            : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
        CheckVk(vkEndCommandBuffer(commands), "vkEndCommandBuffer");
    });

    VkSemaphoreSubmitInfo acquired{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    acquired.semaphore = slot.acquired;
    acquired.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSemaphoreSubmitInfo presentable{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    presentable.semaphore = image.presentable;
    presentable.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    vulkan::NativeSubmit submit{.commands = {&slot.commands, 1},
                                .after = {after.data(), after.size()}};
    if (m_window != nullptr) {
        submit.waits = {&acquired, 1};
        submit.signals = {&presentable, 1};
    }
    slot.done = vulkan::Submit(m_device, submit, vulkan::QueueRole::Display);
    if (m_window == nullptr) {
        return slot.done;
    }

    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &image.presentable;
    present.swapchainCount = 1;
    present.pSwapchains = &m_swapchain;
    present.pImageIndices = &m_index;
    const VkResult result = vulkan::Present(m_device, present);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        m_stale = true;
    } else {
        CheckVk(result, "vkQueuePresentKHR");
    }
    return slot.done;
}

std::vector<uint8_t> VulkanPresenter::ReadPixels(const device::Operation& after) {
    if (m_window != nullptr) {
        throw Error("Only an offscreen viewer reads its pixels back");
    }
    const VkDeviceSize bytes = VkDeviceSize(m_extent.width) * m_extent.height * 4;
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = bytes;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buffer = VK_NULL_HANDLE;
    CheckVk(vkCreateBuffer(m_native.device, &bufferInfo, nullptr, &buffer), "vkCreateBuffer");
    VkMemoryRequirements needs;
    vkGetBufferMemoryRequirements(m_native.device, buffer, &needs);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = needs.size;
    allocation.memoryTypeIndex =
        MemoryType(m_native.physicalDevice, needs.memoryTypeBits,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory memory = VK_NULL_HANDLE;
    CheckVk(vkAllocateMemory(m_native.device, &allocation, nullptr, &memory), "vkAllocateMemory");
    CheckVk(vkBindBufferMemory(m_native.device, buffer, memory, 0), "vkBindBufferMemory");

    const VkCommandPool pool = CreatePool(m_native.device, m_native.queueFamily);
    const VkCommandBuffer commands = AllocateCommands(m_native.device, pool);
    BeginCommands(commands);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {m_extent.width, m_extent.height, 1};
    vkCmdCopyImageToBuffer(commands, m_images[0].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           buffer, 1, &region);
    VkMemoryBarrier2 toHost{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    toHost.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    toHost.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    toHost.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    toHost.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &toHost;
    vkCmdPipelineBarrier2(commands, &dependency);
    CheckVk(vkEndCommandBuffer(commands), "vkEndCommandBuffer");
    vulkan::Submit(m_device, {.commands = {&commands, 1}, .after = {&after, 1}},
                   vulkan::QueueRole::Display)
        .Wait();

    std::vector<uint8_t> pixels(bytes);
    void* mapped = nullptr;
    CheckVk(vkMapMemory(m_native.device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
    std::memcpy(pixels.data(), mapped, bytes);
    vkDestroyCommandPool(m_native.device, pool, nullptr);
    vkDestroyBuffer(m_native.device, buffer, nullptr);
    vkFreeMemory(m_native.device, memory, nullptr);
    return pixels;
}

}

std::unique_ptr<Presenter> CreateVulkanPresenter(device::Device& device, GLFWwindow* window,
                                                 Extent size, bool vsync) {
    return std::make_unique<VulkanPresenter>(device, window, size, vsync);
}

}

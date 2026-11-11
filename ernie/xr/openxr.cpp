// SPDX-License-Identifier: BSD-3-Clause
#include <fmt/format.h>
#include <vulkan/vulkan_core.h>

#define XR_USE_GRAPHICS_API_VULKAN
#ifndef _WIN32
#include <time.h>
#define XR_USE_TIMESPEC
#endif
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ernie/device/vulkan/native.h"
#include "ernie/scene/rotation.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"
#include "ernie/xr/profiles.h"
#include "ernie/xr/runtime.h"

namespace ernie::xr {
namespace {

constexpr int64_t kSrgbFormat = VK_FORMAT_R8G8B8A8_SRGB;
constexpr int64_t kUnormFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr int64_t kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr XrViewConfigurationType kStereo = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
constexpr XrSpaceLocationFlags kLocated =
    XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
constexpr uint32_t kCommandRing = 3;
static_assert(XR_HAND_JOINT_COUNT_EXT == kHandJointCount, "HandJoint follows XrHandJointEXT");

const char* FeatureExtension(Feature feature) {
    switch (feature) {
    case Feature::EyeGaze:
        return XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME;
    case Feature::HandTracking:
        return XR_EXT_HAND_TRACKING_EXTENSION_NAME;
    case Feature::DisplayRefreshRate:
        return XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME;
    case Feature::PerformanceMetrics:
        return XR_META_PERFORMANCE_METRICS_EXTENSION_NAME;
    }
    return "";
}

const char* StateName(XrSessionState state) {
    switch (state) {
    case XR_SESSION_STATE_IDLE:
        return "idle";
    case XR_SESSION_STATE_READY:
        return "ready";
    case XR_SESSION_STATE_SYNCHRONIZED:
        return "synchronized";
    case XR_SESSION_STATE_VISIBLE:
        return "visible";
    case XR_SESSION_STATE_FOCUSED:
        return "focused";
    case XR_SESSION_STATE_STOPPING:
        return "stopping";
    case XR_SESSION_STATE_LOSS_PENDING:
        return "loss pending";
    case XR_SESSION_STATE_EXITING:
        return "exiting";
    default:
        return "unknown";
    }
}

void CopyName(char* destination, size_t size, std::string_view name) {
    const size_t length = std::min(size - 1, name.size());
    std::memcpy(destination, name.data(), length);
    destination[length] = '\0';
}

Pose StagePose(const XrPosef& pose, bool valid) {
    const double half = std::sqrt(0.5);
    const Quatd orientation{pose.orientation.x, pose.orientation.y, pose.orientation.z,
                            pose.orientation.w};
    return {{pose.position.x, -pose.position.z, pose.position.y},
            Multiply({half, 0, 0, half}, orientation),
            valid};
}

Pose StagePose(const XrSpaceLocation& location) {
    return StagePose(location.pose, (location.locationFlags & kLocated) == kLocated);
}

std::string BindingPath(const ComponentBinding& binding) {
    const std::string hand = binding.hand == 0 ? "/user/hand/left" : "/user/hand/right";
    return binding.slot == Slot::Haptic ? hand + "/output/haptic"
                                        : hand + "/input/" + binding.component;
}

struct EyeSwapchain {
    XrSwapchain handle = XR_NULL_HANDLE;
    std::vector<VkImage> images;
};

struct HandSpaces {
    XrPath path = XR_NULL_PATH;
    XrSpace grip = XR_NULL_HANDLE;
    XrSpace aim = XR_NULL_HANDLE;
};

struct DeviceFunctions {
    PFN_vkCreateCommandPool createCommandPool = nullptr;
    PFN_vkDestroyCommandPool destroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers allocateCommandBuffers = nullptr;
    PFN_vkResetCommandBuffer resetCommandBuffer = nullptr;
    PFN_vkBeginCommandBuffer beginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer endCommandBuffer = nullptr;
    PFN_vkCmdPipelineBarrier cmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyBufferToImage cmdCopyBufferToImage = nullptr;
};

struct Recording {
    VkCommandBuffer commands = VK_NULL_HANDLE;
    device::Operation done;
    device::Tensor staged[2];
    device::Tensor kept[2];
};

class OpenXrRuntime final : public Runtime {
  public:
    explicit OpenXrRuntime(const SessionDesc& desc) : m_desc(desc) {
        CreateInstance();
        try {
            FindSystem();
            LoadFunctions();
            CreateActions();
        } catch (...) {
            xrDestroyInstance(m_instance);
            throw;
        }
    }

    ~OpenXrRuntime() override {
        Stop();
        if (m_actionSet != XR_NULL_HANDLE) {
            xrDestroyActionSet(m_actionSet);
        }
        xrDestroyInstance(m_instance);
    }

    void ConfigureDevice(device::DeviceDesc& desc) override {
        auto creator = std::make_shared<device::vulkan::Creator>();
        creator->createInstance = [this](PFN_vkGetInstanceProcAddr get,
                                         const VkInstanceCreateInfo& info) {
            CheckGraphicsRequirements();
            XrVulkanInstanceCreateInfoKHR create{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
            create.systemId = m_system;
            create.pfnGetInstanceProcAddr = get;
            create.vulkanCreateInfo = &info;
            VkInstance instance = VK_NULL_HANDLE;
            VkResult result = VK_SUCCESS;
            Check(m_createVulkanInstance(m_instance, &create, &instance, &result),
                  "xrCreateVulkanInstanceKHR");
            if (result != VK_SUCCESS) {
                throw Error("the OpenXR runtime's vkCreateInstance failed with VkResult {}",
                            int(result));
            }
            return instance;
        };
        creator->pickAdapter = [this](VkInstance instance) { return HeadsetAdapter(instance); };
        creator->createDevice = [this](PFN_vkGetInstanceProcAddr get, VkPhysicalDevice adapter,
                                       const VkDeviceCreateInfo& info) {
            XrVulkanDeviceCreateInfoKHR create{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
            create.systemId = m_system;
            create.pfnGetInstanceProcAddr = get;
            create.vulkanPhysicalDevice = adapter;
            create.vulkanCreateInfo = &info;
            VkDevice device = VK_NULL_HANDLE;
            VkResult result = VK_SUCCESS;
            Check(m_createVulkanDevice(m_instance, &create, &device, &result),
                  "xrCreateVulkanDeviceKHR");
            if (result != VK_SUCCESS) {
                throw Error("the OpenXR runtime's vkCreateDevice failed with VkResult {}",
                            int(result));
            }
            return device;
        };
        desc.backend = device::Backend::Vulkan;
        desc.presentation = true;
        desc.creator = creator;
    }

    void Start(device::Device& device) override {
        if (device.Adapter().backend != device::Backend::Vulkan || !device.Presents()) {
            throw Error("the OpenXR session needs the Vulkan device Session::CreateDevice makes");
        }
        const device::vulkan::Native native = device::vulkan::NativeHandles(device);
        CheckGraphicsRequirements();
        if (HeadsetAdapter(native.instance) != native.physicalDevice) {
            throw Error("the device isn't on the headset's adapter; make it with "
                        "Session::CreateDevice");
        }
        m_device = Ref<device::Device>(&device);
        LoadDeviceFunctions(native);
        XrGraphicsBindingVulkan2KHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
        binding.instance = native.instance;
        binding.physicalDevice = native.physicalDevice;
        binding.device = native.device;
        binding.queueFamilyIndex = native.queueFamily;
        binding.queueIndex = native.queueIndex;
        XrSessionCreateInfo create{XR_TYPE_SESSION_CREATE_INFO};
        create.next = &binding;
        create.systemId = m_system;
        Check(xrCreateSession(m_instance, &create, &m_session), "xrCreateSession");
        CreateSpaces();
        XrSessionActionSetsAttachInfo attach{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
        attach.countActionSets = 1;
        attach.actionSets = &m_actionSet;
        Check(xrAttachSessionActionSets(m_session, &attach), "xrAttachSessionActionSets");
        CreateSwapchains();
        CreateCommands(native);
        ReadRefreshRates();
        CreateHandTrackers();
        StartMetrics();
    }

    void Stop() override {
        for (Recording& recording : m_recordings) {
            recording.done.Wait();
            recording = {};
        }
        if (m_commandPool != VK_NULL_HANDLE) {
            m_vk.destroyCommandPool(m_vkDevice, m_commandPool, nullptr);
            m_commandPool = VK_NULL_HANDLE;
        }
        for (XrHandTrackerEXT& tracker : m_handTrackers) {
            if (tracker != XR_NULL_HANDLE) {
                m_destroyHandTracker(tracker);
                tracker = XR_NULL_HANDLE;
            }
        }
        m_counters.clear();
        for (EyeSwapchain* chains : {m_swapchains, m_depthSwapchains}) {
            for (uint32_t eye = 0; eye < 2; ++eye) {
                if (chains[eye].handle != XR_NULL_HANDLE) {
                    xrDestroySwapchain(chains[eye].handle);
                }
                chains[eye] = {};
            }
        }
        for (HandSpaces& hand : m_hands) {
            for (XrSpace* space : {&hand.grip, &hand.aim}) {
                if (*space != XR_NULL_HANDLE) {
                    xrDestroySpace(*space);
                    *space = XR_NULL_HANDLE;
                }
            }
        }
        for (XrSpace* space : {&m_gazeSpace, &m_viewSpace, &m_space}) {
            if (*space != XR_NULL_HANDLE) {
                xrDestroySpace(*space);
                *space = XR_NULL_HANDLE;
            }
        }
        if (m_session != XR_NULL_HANDLE) {
            xrDestroySession(m_session);
            m_session = XR_NULL_HANDLE;
        }
        m_running = false;
        m_device = {};
    }

    bool Running() override {
        PollEvents();
        return !m_exit;
    }

    void BeginFrame(Frame& frame, FrameClock& clock) override {
        PollEvents();
        while (m_session != XR_NULL_HANDLE && !m_running && !m_exit) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            PollEvents();
        }
        if (!m_running) {
            frame.shouldRender = false;
            frame.state = SessionState(std::min<uint32_t>(m_state, XR_SESSION_STATE_EXITING));
            return;
        }
        XrFrameWaitInfo wait{XR_TYPE_FRAME_WAIT_INFO};
        XrFrameState state{XR_TYPE_FRAME_STATE};
        {
            ERNIE_ZONE("XR/xrWaitFrame");
            Check(xrWaitFrame(m_session, &wait, &state), "xrWaitFrame");
        }
        const double woke = HostSeconds();
        clock.period = double(state.predictedDisplayPeriod) * 1e-9;
        if (m_lastDisplayTime != 0 && state.predictedDisplayPeriod > 0) {
            const int64_t skipped = (state.predictedDisplayTime - m_lastDisplayTime +
                                     state.predictedDisplayPeriod / 2) /
                                        state.predictedDisplayPeriod -
                                    1;
            clock.missed = uint32_t(std::max<int64_t>(skipped, 0));
        }
        clock.display = DisplayHostTime(state.predictedDisplayTime, woke, clock.period);
        clock.estimated = m_convertTime == nullptr;
        XrFrameBeginInfo begin{XR_TYPE_FRAME_BEGIN_INFO};
        Check(xrBeginFrame(m_session, &begin), "xrBeginFrame");
        m_frameBegun = true;
        m_displayTime = state.predictedDisplayTime;
        if (!m_hasRefreshRate && state.predictedDisplayPeriod > 0) {
            m_rate = float(1e9 / double(state.predictedDisplayPeriod));
        }
        frame.displayTime = state.predictedDisplayTime;
        frame.deltaTime =
            double(m_lastDisplayTime != 0 ? state.predictedDisplayTime - m_lastDisplayTime
                                          : state.predictedDisplayPeriod) *
            1e-9;
        m_lastDisplayTime = state.predictedDisplayTime;
        frame.shouldRender = state.shouldRender == XR_TRUE;
        frame.state = SessionState(std::min<uint32_t>(m_state, XR_SESSION_STATE_EXITING));
        LocateViews(frame);
        clock.poses = HostSeconds();
        ReadInputs(frame);
    }

    void EndFrame(const Frame& frame, const EyeImages* eyes) override {
        if (!m_frameBegun) {
            return;
        }
        m_frameBegun = false;
        XrCompositionLayerProjectionView views[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
                                                     {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
        XrCompositionLayerDepthInfoKHR depths[2] = {{XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR},
                                                    {XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR}};
        XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        std::vector<const XrCompositionLayerBaseHeader*> layers;
        if (eyes != nullptr && frame.shouldRender && m_viewsValid) {
            const bool withDepth = m_depthLayer && !eyes->depth.Empty();
            const XrExtent2Di rect = CopyToSwapchains(*eyes, withDepth);
            for (uint32_t eye = 0; eye < 2; ++eye) {
                if (withDepth) {
                    XrCompositionLayerDepthInfoKHR& info = depths[eye];
                    info.subImage.swapchain = m_depthSwapchains[eye].handle;
                    info.subImage.imageRect = {{0, 0}, rect};
                    info.minDepth = 0;
                    info.maxDepth = 1;
                    info.nearZ = m_desc.nearPlane;
                    info.farZ = m_desc.farPlane;
                    views[eye].next = &info;
                }
                views[eye].pose = m_views[eye].pose;
                views[eye].fov = m_views[eye].fov;
                views[eye].subImage.swapchain = m_swapchains[eye].handle;
                views[eye].subImage.imageRect = {{0, 0}, rect};
            }
            layer.space = m_space;
            if (m_blendMode == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND) {
                layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT |
                                   XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT;
            }
            layer.viewCount = 2;
            layer.views = views;
            layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&layer));
        }
        SendHaptics(frame);
        XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO};
        end.displayTime = m_displayTime;
        end.environmentBlendMode = m_blendMode;
        end.layerCount = uint32_t(layers.size());
        end.layers = layers.data();
        XrResult result = XR_SUCCESS;
        {
            ERNIE_ZONE("XR/xrEndFrame");
            device::vulkan::Exclusive(
                *m_device, [&](VkQueue) { result = xrEndFrame(m_session, &end); },
                device::vulkan::QueueRole::Render);
        }
        Check(result, "xrEndFrame");
    }

    void RequestExit() override {
        if (m_session != XR_NULL_HANDLE && m_running) {
            Check(xrRequestExitSession(m_session), "xrRequestExitSession");
        } else {
            m_exit = true;
        }
    }

    bool DepthLayer() const override {
        return m_depthLayer;
    }

    BlendMode Blend() const override {
        return m_blend;
    }

    bool SetRefreshRate(float hz) override {
        if (!m_hasRefreshRate || m_session == XR_NULL_HANDLE ||
            std::ranges::find(m_rates, hz) == m_rates.end()) {
            return false;
        }
        return XR_SUCCEEDED(m_requestRefreshRate(m_session, hz));
    }

    Uint2 Recommended() const override {
        return m_recommended;
    }

    uint32_t Features() const override {
        return m_features;
    }

    void FillStatus(SessionStatus& status) const override {
        status.runtime = m_runtimeName;
        status.system = m_systemName;
        status.state = StateName(m_state);
        status.refreshRate = m_rate;
        status.refreshRates = m_rates;
        status.playArea = m_playArea;
    }

    RuntimeMetrics Metrics(const SessionStats&) override {
        RuntimeMetrics metrics;
        if (m_counters.empty() || m_session == XR_NULL_HANDLE) {
            return metrics;
        }
        metrics.available = true;
        for (const auto& [counter, path] : m_counters) {
            XrPerformanceMetricsCounterMETA value{XR_TYPE_PERFORMANCE_METRICS_COUNTER_META};
            if (XR_FAILED(m_queryCounter(m_session, path, &value)) ||
                (value.counterFlags & XR_PERFORMANCE_METRICS_COUNTER_ANY_VALUE_VALID_BIT_META) ==
                    0) {
                continue;
            }
            const bool isFloat = (value.counterFlags &
                                  XR_PERFORMANCE_METRICS_COUNTER_FLOAT_VALUE_VALID_BIT_META) != 0;
            const double number = isFloat ? value.floatValue : double(value.uintValue);
            const double seconds =
                value.counterUnit == XR_PERFORMANCE_METRICS_COUNTER_UNIT_MILLISECONDS_META
                    ? number * 1e-3
                    : number;
            const float fraction =
                float(value.counterUnit == XR_PERFORMANCE_METRICS_COUNTER_UNIT_PERCENTAGE_META
                          ? number / 100
                          : number);
            switch (counter) {
            case Counter::AppCpu:
                metrics.appCpu = seconds;
                break;
            case Counter::AppGpu:
                metrics.appGpu = seconds;
                break;
            case Counter::AppLatency:
                metrics.motionToPhoton = seconds;
                break;
            case Counter::CompositorCpu:
                metrics.compositorCpu = seconds;
                break;
            case Counter::CompositorGpu:
                metrics.compositorGpu = seconds;
                break;
            case Counter::Dropped:
                metrics.droppedFrames = uint32_t(number);
                break;
            case Counter::CpuUse:
                metrics.cpuUtilization = fraction;
                break;
            case Counter::GpuUse:
                metrics.gpuUtilization = fraction;
                break;
            }
        }
        return metrics;
    }

  private:
    enum class Counter {
        AppCpu,
        AppGpu,
        AppLatency,
        CompositorCpu,
        CompositorGpu,
        Dropped,
        CpuUse,
        GpuUse
    };

    void Check(XrResult result, const char* call) const {
        if (XR_SUCCEEDED(result)) {
            return;
        }
        char name[XR_MAX_RESULT_STRING_SIZE] = {};
        if (m_instance == XR_NULL_HANDLE || XR_FAILED(xrResultToString(m_instance, result, name))) {
            std::snprintf(name, sizeof(name), "XrResult %d", int(result));
        }
        throw Error("{} failed: {}", call, name);
    }

    void CreateInstance() {
        uint32_t count = 0;
        const XrResult listed = xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr);
        if (XR_FAILED(listed)) {
            throw UnsupportedError(
                "no OpenXR runtime is active (XrResult {}); start one, such as SteamVR, Meta "
                "Quest Link or Monado, or set ERNIE_XR=sim for the simulated headset",
                int(listed));
        }
        std::vector<XrExtensionProperties> offered(count, {XR_TYPE_EXTENSION_PROPERTIES});
        Check(xrEnumerateInstanceExtensionProperties(nullptr, count, &count, offered.data()),
              "xrEnumerateInstanceExtensionProperties");
        const auto has = [&](std::string_view name) {
            return std::ranges::any_of(offered, [&](const XrExtensionProperties& extension) {
                return name == extension.extensionName;
            });
        };
        if (!has(XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME)) {
            throw UnsupportedError("the OpenXR runtime lacks {}",
                                   XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME);
        }
        const auto enable = [&](std::string_view name) {
            if (!has(name)) {
                return false;
            }
            if (std::ranges::find(m_enabled, name) == m_enabled.end()) {
                m_enabled.emplace_back(name);
            }
            return true;
        };
        enable(XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME);
        std::vector<Feature> features = m_desc.features;
        if (m_desc.refreshRate > 0) {
            features.push_back(Feature::DisplayRefreshRate);
        }
        for (const Feature feature : features) {
            if (enable(FeatureExtension(feature))) {
                m_features |= 1u << uint32_t(feature);
            } else {
                Log(LogLevel::Info, "the OpenXR runtime lacks {}; going on without it",
                    FeatureExtension(feature));
            }
        }
        if (m_desc.depthLayer) {
            m_depthLayer = enable(XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME);
        }
        for (const InteractionProfile& profile : InteractionProfiles()) {
            if (!profile.extension.empty()) {
                enable(profile.extension);
            }
        }
#ifndef _WIN32
        m_hasTimespec = enable(XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME);
#endif
        std::vector<const char*> names;
        for (const std::string& name : m_enabled) {
            names.push_back(name.c_str());
        }
        XrInstanceCreateInfo create{XR_TYPE_INSTANCE_CREATE_INFO};
        CopyName(create.applicationInfo.applicationName, XR_MAX_APPLICATION_NAME_SIZE,
                 m_desc.application);
        CopyName(create.applicationInfo.engineName, XR_MAX_ENGINE_NAME_SIZE, "Ernie");
        create.applicationInfo.apiVersion = XR_API_VERSION_1_0;
        create.enabledExtensionCount = uint32_t(names.size());
        create.enabledExtensionNames = names.data();
        const XrResult created = xrCreateInstance(&create, &m_instance);
        if (created == XR_ERROR_RUNTIME_UNAVAILABLE || created == XR_ERROR_RUNTIME_FAILURE) {
            throw UnsupportedError("the OpenXR runtime didn't start (XrResult {}); start its "
                                   "service, or set ERNIE_XR=sim for the simulated headset",
                                   int(created));
        }
        Check(created, "xrCreateInstance");
        XrInstanceProperties properties{XR_TYPE_INSTANCE_PROPERTIES};
        Check(xrGetInstanceProperties(m_instance, &properties), "xrGetInstanceProperties");
        m_runtimeName = fmt::format("{} {}.{}.{}", properties.runtimeName,
                                    XR_VERSION_MAJOR(properties.runtimeVersion),
                                    XR_VERSION_MINOR(properties.runtimeVersion),
                                    XR_VERSION_PATCH(properties.runtimeVersion));
    }

    void FindSystem() {
        XrSystemGetInfo get{XR_TYPE_SYSTEM_GET_INFO};
        get.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        const XrResult result = xrGetSystem(m_instance, &get, &m_system);
        if (result == XR_ERROR_FORM_FACTOR_UNAVAILABLE) {
            throw UnsupportedError("the OpenXR runtime {} has no headset connected", m_runtimeName);
        }
        Check(result, "xrGetSystem");
        XrSystemEyeGazeInteractionPropertiesEXT gaze{
            XR_TYPE_SYSTEM_EYE_GAZE_INTERACTION_PROPERTIES_EXT};
        XrSystemHandTrackingPropertiesEXT hands{XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT};
        XrSystemProperties properties{XR_TYPE_SYSTEM_PROPERTIES};
        if (Has(Feature::EyeGaze)) {
            gaze.next = properties.next;
            properties.next = &gaze;
        }
        if (Has(Feature::HandTracking)) {
            hands.next = properties.next;
            properties.next = &hands;
        }
        Check(xrGetSystemProperties(m_instance, m_system, &properties), "xrGetSystemProperties");
        m_systemName = properties.systemName;
        if (Has(Feature::EyeGaze) && gaze.supportsEyeGazeInteraction != XR_TRUE) {
            m_features &= ~(1u << uint32_t(Feature::EyeGaze));
        }
        if (Has(Feature::HandTracking) && hands.supportsHandTracking != XR_TRUE) {
            m_features &= ~(1u << uint32_t(Feature::HandTracking));
        }
        uint32_t count = 0;
        Check(xrEnumerateViewConfigurationViews(m_instance, m_system, kStereo, 0, &count, nullptr),
              "xrEnumerateViewConfigurationViews");
        std::vector<XrViewConfigurationView> views(count, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
        Check(xrEnumerateViewConfigurationViews(m_instance, m_system, kStereo, count, &count,
                                                views.data()),
              "xrEnumerateViewConfigurationViews");
        if (count != 2) {
            throw UnsupportedError("the headset has no stereo view configuration");
        }
        m_recommended = {views[0].recommendedImageRectWidth, views[0].recommendedImageRectHeight};
        const float scale = std::clamp(m_desc.resolutionScale, 0.1f, 2.0f);
        m_swapchainSize = {std::clamp(uint32_t(std::lround(float(m_recommended.x) * scale)), 1u,
                                      views[0].maxImageRectWidth),
                           std::clamp(uint32_t(std::lround(float(m_recommended.y) * scale)), 1u,
                                      views[0].maxImageRectHeight)};
        ChooseBlendMode();
    }

    void ChooseBlendMode() {
        uint32_t count = 0;
        Check(xrEnumerateEnvironmentBlendModes(m_instance, m_system, kStereo, 0, &count, nullptr),
              "xrEnumerateEnvironmentBlendModes");
        std::vector<XrEnvironmentBlendMode> modes(count);
        Check(xrEnumerateEnvironmentBlendModes(m_instance, m_system, kStereo, count, &count,
                                               modes.data()),
              "xrEnumerateEnvironmentBlendModes");
        const XrEnvironmentBlendMode wanted =
            m_desc.blendMode == BlendMode::Additive     ? XR_ENVIRONMENT_BLEND_MODE_ADDITIVE
            : m_desc.blendMode == BlendMode::AlphaBlend ? XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND
                                                        : XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        if (std::ranges::find(modes, wanted) != modes.end()) {
            m_blendMode = wanted;
        } else {
            if (wanted != XR_ENVIRONMENT_BLEND_MODE_OPAQUE) {
                Log(LogLevel::Warning, "the headset doesn't offer that blend mode, so it's opaque");
            }
            m_blendMode = modes.empty() ? XR_ENVIRONMENT_BLEND_MODE_OPAQUE : modes.front();
        }
        m_blend = m_blendMode == XR_ENVIRONMENT_BLEND_MODE_ADDITIVE      ? BlendMode::Additive
                  : m_blendMode == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND ? BlendMode::AlphaBlend
                                                                         : BlendMode::Opaque;
    }

    template <class Function> void Load(const char* name, Function& function) {
        Check(xrGetInstanceProcAddr(m_instance, name,
                                    reinterpret_cast<PFN_xrVoidFunction*>(&function)),
              name);
    }

    void LoadFunctions() {
        Load("xrGetVulkanGraphicsRequirements2KHR", m_graphicsRequirements);
        Load("xrCreateVulkanInstanceKHR", m_createVulkanInstance);
        Load("xrCreateVulkanDeviceKHR", m_createVulkanDevice);
        Load("xrGetVulkanGraphicsDevice2KHR", m_graphicsDevice);
        if (Has(Feature::DisplayRefreshRate)) {
            Load("xrEnumerateDisplayRefreshRatesFB", m_enumerateRefreshRates);
            Load("xrGetDisplayRefreshRateFB", m_getRefreshRate);
            Load("xrRequestDisplayRefreshRateFB", m_requestRefreshRate);
        }
        if (Has(Feature::HandTracking)) {
            Load("xrCreateHandTrackerEXT", m_createHandTracker);
            Load("xrDestroyHandTrackerEXT", m_destroyHandTracker);
            Load("xrLocateHandJointsEXT", m_locateHandJoints);
        }
        if (Has(Feature::PerformanceMetrics)) {
            Load("xrEnumeratePerformanceMetricsCounterPathsMETA", m_enumerateCounters);
            Load("xrSetPerformanceMetricsStateMETA", m_setMetricsState);
            Load("xrQueryPerformanceMetricsCounterMETA", m_queryCounter);
        }
#ifndef _WIN32
        if (m_hasTimespec) {
            Load("xrConvertTimeToTimespecTimeKHR", m_convertTime);
        }
#endif
    }

    void LoadDeviceFunctions(const device::vulkan::Native& native) {
        const auto getDevice = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            native.getInstanceProcAddr(native.instance, "vkGetDeviceProcAddr"));
        m_vkDevice = native.device;
        const auto load = [&](const char* name, auto& function) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
                getDevice(native.device, name));
            if (function == nullptr) {
                throw Error("the Vulkan device has no {}", name);
            }
        };
        load("vkCreateCommandPool", m_vk.createCommandPool);
        load("vkDestroyCommandPool", m_vk.destroyCommandPool);
        load("vkAllocateCommandBuffers", m_vk.allocateCommandBuffers);
        load("vkResetCommandBuffer", m_vk.resetCommandBuffer);
        load("vkBeginCommandBuffer", m_vk.beginCommandBuffer);
        load("vkEndCommandBuffer", m_vk.endCommandBuffer);
        load("vkCmdPipelineBarrier", m_vk.cmdPipelineBarrier);
        load("vkCmdCopyBufferToImage", m_vk.cmdCopyBufferToImage);
    }

    double DisplayHostTime(XrTime display, double woke, double period) const {
#ifndef _WIN32
        timespec converted{};
        if (m_convertTime != nullptr &&
            XR_SUCCEEDED(m_convertTime(m_instance, display, &converted))) {
            return double(converted.tv_sec) + double(converted.tv_nsec) * 1e-9;
        }
#endif
        (void)display;
        return woke + 2 * period;
    }

    bool Has(Feature feature) const {
        return (m_features >> uint32_t(feature) & 1u) != 0;
    }

    void CheckGraphicsRequirements() {
        if (m_graphicsChecked) {
            return;
        }
        XrGraphicsRequirementsVulkan2KHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
        Check(m_graphicsRequirements(m_instance, m_system, &requirements),
              "xrGetVulkanGraphicsRequirements2KHR");
        if (requirements.minApiVersionSupported > XR_MAKE_VERSION(1, 3, 0)) {
            throw UnsupportedError("the OpenXR runtime needs a Vulkan newer than Ernie's 1.3");
        }
        m_graphicsChecked = true;
    }

    VkPhysicalDevice HeadsetAdapter(VkInstance instance) {
        XrVulkanGraphicsDeviceGetInfoKHR get{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
        get.systemId = m_system;
        get.vulkanInstance = instance;
        VkPhysicalDevice adapter = VK_NULL_HANDLE;
        Check(m_graphicsDevice(m_instance, &get, &adapter), "xrGetVulkanGraphicsDevice2KHR");
        return adapter;
    }

    XrPath Path(const std::string& text) const {
        XrPath path = XR_NULL_PATH;
        Check(xrStringToPath(m_instance, text.c_str(), &path), "xrStringToPath");
        return path;
    }

    XrAction CreateAction(const char* name, const char* label, XrActionType type, bool hands) {
        XrActionCreateInfo create{XR_TYPE_ACTION_CREATE_INFO};
        CopyName(create.actionName, XR_MAX_ACTION_NAME_SIZE, name);
        CopyName(create.localizedActionName, XR_MAX_LOCALIZED_ACTION_NAME_SIZE, label);
        create.actionType = type;
        XrPath paths[2] = {m_hands[0].path, m_hands[1].path};
        if (hands) {
            create.countSubactionPaths = 2;
            create.subactionPaths = paths;
        }
        XrAction action = XR_NULL_HANDLE;
        Check(xrCreateAction(m_actionSet, &create, &action), "xrCreateAction");
        return action;
    }

    void CreateActions() {
        XrActionSetCreateInfo create{XR_TYPE_ACTION_SET_CREATE_INFO};
        CopyName(create.actionSetName, XR_MAX_ACTION_SET_NAME_SIZE, "ernie");
        CopyName(create.localizedActionSetName, XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE, "Ernie");
        Check(xrCreateActionSet(m_instance, &create, &m_actionSet), "xrCreateActionSet");
        m_hands[0].path = Path("/user/hand/left");
        m_hands[1].path = Path("/user/hand/right");
        m_actions[uint32_t(Slot::GripPose)] =
            CreateAction("grip_pose", "Grip pose", XR_ACTION_TYPE_POSE_INPUT, true);
        m_actions[uint32_t(Slot::AimPose)] =
            CreateAction("aim_pose", "Aim pose", XR_ACTION_TYPE_POSE_INPUT, true);
        m_actions[uint32_t(Slot::Trigger)] =
            CreateAction("trigger", "Trigger", XR_ACTION_TYPE_FLOAT_INPUT, true);
        m_actions[uint32_t(Slot::Squeeze)] =
            CreateAction("grip", "Grip", XR_ACTION_TYPE_FLOAT_INPUT, true);
        m_actions[uint32_t(Slot::Stick)] =
            CreateAction("thumbstick", "Thumbstick", XR_ACTION_TYPE_VECTOR2F_INPUT, true);
        m_actions[uint32_t(Slot::StickClick)] = CreateAction("thumbstick_click", "Thumbstick click",
                                                             XR_ACTION_TYPE_BOOLEAN_INPUT, true);
        m_actions[uint32_t(Slot::Primary)] =
            CreateAction("primary", "Primary button", XR_ACTION_TYPE_BOOLEAN_INPUT, true);
        m_actions[uint32_t(Slot::Secondary)] =
            CreateAction("secondary", "Secondary button", XR_ACTION_TYPE_BOOLEAN_INPUT, true);
        m_actions[uint32_t(Slot::Menu)] =
            CreateAction("menu", "Menu", XR_ACTION_TYPE_BOOLEAN_INPUT, true);
        m_actions[uint32_t(Slot::Haptic)] =
            CreateAction("haptic", "Vibration", XR_ACTION_TYPE_VIBRATION_OUTPUT, true);
        if (Has(Feature::EyeGaze)) {
            m_gaze = CreateAction("gaze", "Eye gaze", XR_ACTION_TYPE_POSE_INPUT, false);
        }
        for (const InteractionProfile& profile : InteractionProfiles()) {
            if (!profile.extension.empty() &&
                std::ranges::find(m_enabled, profile.extension) == m_enabled.end()) {
                continue;
            }
            std::vector<std::pair<XrAction, std::string>> bindings;
            for (const ComponentBinding& binding : profile.bindings) {
                bindings.emplace_back(m_actions[uint32_t(binding.slot)], BindingPath(binding));
            }
            Suggest(profile.path, bindings);
        }
        if (Has(Feature::EyeGaze)) {
            Suggest("/interaction_profiles/ext/eye_gaze_interaction",
                    {{m_gaze, "/user/eyes_ext/input/gaze_ext/pose"}});
        }
    }

    void Suggest(const std::string& profile,
                 const std::vector<std::pair<XrAction, std::string>>& bindings) {
        std::vector<XrActionSuggestedBinding> suggested;
        for (const auto& [action, path] : bindings) {
            suggested.push_back({action, Path(path)});
        }
        XrInteractionProfileSuggestedBinding suggestion{
            XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        suggestion.interactionProfile = Path(profile);
        suggestion.countSuggestedBindings = uint32_t(suggested.size());
        suggestion.suggestedBindings = suggested.data();
        const XrResult result = xrSuggestInteractionProfileBindings(m_instance, &suggestion);
        if (XR_FAILED(result)) {
            Log(LogLevel::Warning, "the OpenXR runtime refused bindings for {} (XrResult {})",
                profile, int(result));
        }
    }

    XrSpace CreateReferenceSpace(XrReferenceSpaceType type) {
        XrReferenceSpaceCreateInfo create{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
        create.referenceSpaceType = type;
        create.poseInReferenceSpace.orientation.w = 1;
        XrSpace space = XR_NULL_HANDLE;
        Check(xrCreateReferenceSpace(m_session, &create, &space), "xrCreateReferenceSpace");
        return space;
    }

    XrSpace CreateActionSpace(XrAction action, XrPath hand) {
        XrActionSpaceCreateInfo create{XR_TYPE_ACTION_SPACE_CREATE_INFO};
        create.action = action;
        create.subactionPath = hand;
        create.poseInActionSpace.orientation.w = 1;
        XrSpace space = XR_NULL_HANDLE;
        Check(xrCreateActionSpace(m_session, &create, &space), "xrCreateActionSpace");
        return space;
    }

    void CreateSpaces() {
        uint32_t count = 0;
        Check(xrEnumerateReferenceSpaces(m_session, 0, &count, nullptr),
              "xrEnumerateReferenceSpaces");
        std::vector<XrReferenceSpaceType> types(count);
        Check(xrEnumerateReferenceSpaces(m_session, count, &count, types.data()),
              "xrEnumerateReferenceSpaces");
        const bool wantsStage = m_desc.referenceSpace == ReferenceSpace::Stage;
        const bool stage =
            wantsStage && std::ranges::find(types, XR_REFERENCE_SPACE_TYPE_STAGE) != types.end();
        if (wantsStage && !stage) {
            Log(LogLevel::Warning, "the headset has no stage space, so poses are in its local one");
        }
        m_space = CreateReferenceSpace(stage ? XR_REFERENCE_SPACE_TYPE_STAGE
                                             : XR_REFERENCE_SPACE_TYPE_LOCAL);
        XrExtent2Df bounds{};
        if (stage && XR_SUCCEEDED(xrGetReferenceSpaceBoundsRect(
                         m_session, XR_REFERENCE_SPACE_TYPE_STAGE, &bounds))) {
            m_playArea = {bounds.width, bounds.height};
        }
        m_viewSpace = CreateReferenceSpace(XR_REFERENCE_SPACE_TYPE_VIEW);
        for (HandSpaces& hand : m_hands) {
            hand.grip = CreateActionSpace(m_actions[uint32_t(Slot::GripPose)], hand.path);
            hand.aim = CreateActionSpace(m_actions[uint32_t(Slot::AimPose)], hand.path);
        }
        if (Has(Feature::EyeGaze)) {
            m_gazeSpace = CreateActionSpace(m_gaze, XR_NULL_PATH);
        }
    }

    void CreateSwapchains() {
        uint32_t count = 0;
        Check(xrEnumerateSwapchainFormats(m_session, 0, &count, nullptr),
              "xrEnumerateSwapchainFormats");
        std::vector<int64_t> formats(count);
        Check(xrEnumerateSwapchainFormats(m_session, count, &count, formats.data()),
              "xrEnumerateSwapchainFormats");
        const auto offered = [&](int64_t format) {
            return std::ranges::find(formats, format) != formats.end();
        };
        const int64_t format = offered(kSrgbFormat) ? kSrgbFormat : kUnormFormat;
        if (!offered(format)) {
            throw UnsupportedError("the OpenXR runtime offers no RGBA8 swapchain");
        }
        if (format == kUnormFormat) {
            Log(LogLevel::Warning,
                "the OpenXR runtime offers no sRGB swapchain, so colors show too bright");
        }
        for (EyeSwapchain& swapchain : m_swapchains) {
            swapchain = CreateSwapchain(format, XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT |
                                                    XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT);
        }
        if (m_depthLayer && !offered(kDepthFormat)) {
            Log(LogLevel::Warning,
                "the OpenXR runtime offers no D32 depth swapchain, so frames go without depth");
            m_depthLayer = false;
        }
        if (m_depthLayer) {
            for (EyeSwapchain& swapchain : m_depthSwapchains) {
                swapchain = CreateSwapchain(kDepthFormat,
                                            XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT |
                                                XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
            }
        }
    }

    EyeSwapchain CreateSwapchain(int64_t format, XrSwapchainUsageFlags usage) {
        EyeSwapchain swapchain;
        XrSwapchainCreateInfo create{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        create.usageFlags = usage;
        create.format = format;
        create.sampleCount = 1;
        create.width = m_swapchainSize.x;
        create.height = m_swapchainSize.y;
        create.faceCount = 1;
        create.arraySize = 1;
        create.mipCount = 1;
        Check(xrCreateSwapchain(m_session, &create, &swapchain.handle), "xrCreateSwapchain");
        uint32_t images = 0;
        Check(xrEnumerateSwapchainImages(swapchain.handle, 0, &images, nullptr),
              "xrEnumerateSwapchainImages");
        std::vector<XrSwapchainImageVulkan2KHR> vulkan(images,
                                                       {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
        Check(xrEnumerateSwapchainImages(
                  swapchain.handle, images, &images,
                  reinterpret_cast<XrSwapchainImageBaseHeader*>(vulkan.data())),
              "xrEnumerateSwapchainImages");
        for (const XrSwapchainImageVulkan2KHR& image : vulkan) {
            swapchain.images.push_back(image.image);
        }
        return swapchain;
    }

    void CreateCommands(const device::vulkan::Native& native) {
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = native.queueFamily;
        if (m_vk.createCommandPool(native.device, &pool, nullptr, &m_commandPool) != VK_SUCCESS) {
            throw Error("vkCreateCommandPool failed for the OpenXR swapchain copies");
        }
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = m_commandPool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = kCommandRing;
        VkCommandBuffer buffers[kCommandRing] = {};
        if (m_vk.allocateCommandBuffers(native.device, &allocate, buffers) != VK_SUCCESS) {
            throw Error("vkAllocateCommandBuffers failed for the OpenXR swapchain copies");
        }
        for (uint32_t i = 0; i < kCommandRing; ++i) {
            m_recordings[i].commands = buffers[i];
        }
    }

    device::Tensor Stage(const device::Tensor& source, device::Tensor& staged,
                         std::vector<device::Operation>& after, const device::Operation& ready) {
        if (&source.GetDevice() == m_device.Get()) {
            return source;
        }
        const std::vector<int64_t> shape(source.Shape().begin(), source.Shape().end());
        if (staged.Empty() || !std::ranges::equal(staged.Shape(), shape)) {
            staged = m_device->CreateTensor(source.Type(), shape, "xr staged eyes");
        }
        after.push_back(m_device->Copy(source, staged, ready));
        return staged;
    }

    void Transition(VkCommandBuffer commands, VkImage image, VkImageAspectFlags aspect,
                    VkImageLayout from, VkImageLayout to) const {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = from == VK_IMAGE_LAYOUT_UNDEFINED ? VK_ACCESS_MEMORY_WRITE_BIT
                                                                  : VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = from == VK_IMAGE_LAYOUT_UNDEFINED ? VK_ACCESS_TRANSFER_WRITE_BIT
                                                                  : VK_ACCESS_MEMORY_READ_BIT;
        barrier.oldLayout = from;
        barrier.newLayout = to;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {aspect, 0, 1, 0, 1};
        m_vk.cmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                &barrier);
    }

    void RecordCopy(VkCommandBuffer commands, const device::Tensor& eyes, VkImage image,
                    uint32_t eye, VkImageAspectFlags aspect, VkImageLayout final,
                    VkExtent2D rect) const {
        const uint32_t height = uint32_t(eyes.Shape()[1]);
        const uint32_t width = uint32_t(eyes.Shape()[2]);
        constexpr uint64_t texel = 4;
        Transition(commands, image, aspect, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy copy{};
        copy.bufferOffset =
            VkDeviceSize(eyes.Offset()) + VkDeviceSize(eye) * width * height * texel;
        copy.bufferRowLength = width;
        copy.bufferImageHeight = height;
        copy.imageSubresource = {aspect, 0, 0, 1};
        copy.imageExtent = {rect.width, rect.height, 1};
        m_vk.cmdCopyBufferToImage(commands, device::vulkan::NativeBuffer(eyes), image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        Transition(commands, image, aspect, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, final);
    }

    std::array<uint32_t, 2> Acquire(const EyeSwapchain (&chains)[2]) const {
        std::array<uint32_t, 2> indices{};
        for (uint32_t eye = 0; eye < 2; ++eye) {
            XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
            Check(xrAcquireSwapchainImage(chains[eye].handle, &acquire, &indices[eye]),
                  "xrAcquireSwapchainImage");
            XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
            wait.timeout = XR_INFINITE_DURATION;
            Check(xrWaitSwapchainImage(chains[eye].handle, &wait), "xrWaitSwapchainImage");
        }
        return indices;
    }

    void Release(const EyeSwapchain (&chains)[2]) const {
        for (uint32_t eye = 0; eye < 2; ++eye) {
            XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            XrResult result = XR_SUCCESS;
            device::vulkan::Exclusive(
                *m_device,
                [&](VkQueue) { result = xrReleaseSwapchainImage(chains[eye].handle, &release); },
                device::vulkan::QueueRole::Render);
            Check(result, "xrReleaseSwapchainImage");
        }
    }

    XrExtent2Di CopyToSwapchains(const EyeImages& eyes, bool withDepth) {
        ERNIE_ZONE("XR/Swapchain copy");
        const Span<const int64_t> shape = eyes.color.Shape();
        const VkExtent2D rect{std::min(uint32_t(shape[2]), m_swapchainSize.x),
                              std::min(uint32_t(shape[1]), m_swapchainSize.y)};
        Recording& recording = m_recordings[m_nextRecording];
        m_nextRecording = (m_nextRecording + 1) % kCommandRing;
        recording.done.Wait();
        std::vector<device::Operation> after = {eyes.ready};
        const device::Tensor color = Stage(eyes.color, recording.staged[0], after, eyes.ready);
        const device::Tensor depth = withDepth
                                         ? Stage(eyes.depth, recording.staged[1], after, eyes.ready)
                                         : device::Tensor();
        const std::array<uint32_t, 2> colorImages = Acquire(m_swapchains);
        std::array<uint32_t, 2> depthImages{};
        if (withDepth) {
            depthImages = Acquire(m_depthSwapchains);
        }
        VkCommandBuffer commands = recording.commands;
        m_vk.resetCommandBuffer(commands, 0);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        m_vk.beginCommandBuffer(commands, &begin);
        for (uint32_t eye = 0; eye < 2; ++eye) {
            RecordCopy(commands, color, m_swapchains[eye].images[colorImages[eye]], eye,
                       VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, rect);
            if (withDepth) {
                RecordCopy(commands, depth, m_depthSwapchains[eye].images[depthImages[eye]], eye,
                           VK_IMAGE_ASPECT_DEPTH_BIT,
                           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, rect);
            }
        }
        m_vk.endCommandBuffer(commands);
        recording.done =
            device::vulkan::Submit(*m_device, {.commands = {&commands, 1}, .after = after},
                                   device::vulkan::QueueRole::Render);
        recording.kept[0] = color;
        recording.kept[1] = depth;
        Release(m_swapchains);
        if (withDepth) {
            Release(m_depthSwapchains);
        }
        return {int32_t(rect.width), int32_t(rect.height)};
    }

    void ReadRefreshRates() {
        if (!Has(Feature::DisplayRefreshRate)) {
            return;
        }
        uint32_t count = 0;
        Check(m_enumerateRefreshRates(m_session, 0, &count, nullptr),
              "xrEnumerateDisplayRefreshRatesFB");
        m_rates.resize(count);
        Check(m_enumerateRefreshRates(m_session, count, &count, m_rates.data()),
              "xrEnumerateDisplayRefreshRatesFB");
        Check(m_getRefreshRate(m_session, &m_rate), "xrGetDisplayRefreshRateFB");
        m_hasRefreshRate = true;
        const float wanted = m_desc.refreshRate;
        if (wanted > 0 && wanted != m_rate) {
            if (std::ranges::find(m_rates, wanted) == m_rates.end()) {
                Log(LogLevel::Warning, "the headset has no {} Hz rate, so it keeps {} Hz", wanted,
                    m_rate);
            } else {
                Check(m_requestRefreshRate(m_session, wanted), "xrRequestDisplayRefreshRateFB");
            }
        }
    }

    void CreateHandTrackers() {
        if (!Has(Feature::HandTracking)) {
            return;
        }
        for (uint32_t side = 0; side < 2; ++side) {
            XrHandTrackerCreateInfoEXT create{XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT};
            create.hand = side == 0 ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT;
            create.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;
            Check(m_createHandTracker(m_session, &create, &m_handTrackers[side]),
                  "xrCreateHandTrackerEXT");
        }
    }

    void LocateHands(Frame& frame) {
        if (!Has(Feature::HandTracking)) {
            return;
        }
        for (uint32_t side = 0; side < 2; ++side) {
            XrHandJointLocationEXT joints[XR_HAND_JOINT_COUNT_EXT];
            XrHandJointLocationsEXT locations{XR_TYPE_HAND_JOINT_LOCATIONS_EXT};
            locations.jointCount = XR_HAND_JOINT_COUNT_EXT;
            locations.jointLocations = joints;
            XrHandJointsLocateInfoEXT locate{XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT};
            locate.baseSpace = m_space;
            locate.time = m_displayTime;
            Check(m_locateHandJoints(m_handTrackers[side], &locate, &locations),
                  "xrLocateHandJointsEXT");
            HandJoints& hand = frame.hands[side];
            hand.active = locations.isActive == XR_TRUE;
            for (uint32_t joint = 0; joint < kHandJointCount; ++joint) {
                const XrHandJointLocationEXT& location = joints[joint];
                hand.joints[joint].pose = StagePose(
                    location.pose, hand.active && (location.locationFlags & kLocated) == kLocated);
                hand.joints[joint].radius = location.radius;
            }
        }
    }

    void StartMetrics() {
        if (!Has(Feature::PerformanceMetrics)) {
            return;
        }
        XrPerformanceMetricsStateMETA state{XR_TYPE_PERFORMANCE_METRICS_STATE_META};
        state.enabled = XR_TRUE;
        Check(m_setMetricsState(m_session, &state), "xrSetPerformanceMetricsStateMETA");
        uint32_t count = 0;
        Check(m_enumerateCounters(m_instance, 0, &count, nullptr),
              "xrEnumeratePerformanceMetricsCounterPathsMETA");
        std::vector<XrPath> paths(count);
        Check(m_enumerateCounters(m_instance, count, &count, paths.data()),
              "xrEnumeratePerformanceMetricsCounterPathsMETA");
        const std::pair<Counter, const char*> known[] = {
            {Counter::AppCpu, "/perfmetrics_meta/app/cpu_frametime"},
            {Counter::AppGpu, "/perfmetrics_meta/app/gpu_frametime"},
            {Counter::AppLatency, "/perfmetrics_meta/app/motion_to_photon_latency"},
            {Counter::CompositorCpu, "/perfmetrics_meta/compositor/cpu_frametime"},
            {Counter::CompositorGpu, "/perfmetrics_meta/compositor/gpu_frametime"},
            {Counter::Dropped, "/perfmetrics_meta/compositor/dropped_frame_count"},
            {Counter::CpuUse, "/perfmetrics_meta/device/cpu_utilization_average"},
            {Counter::GpuUse, "/perfmetrics_meta/device/gpu_utilization"},
        };
        for (const auto& [counter, name] : known) {
            const XrPath path = Path(name);
            if (std::ranges::find(paths, path) != paths.end()) {
                m_counters.emplace_back(counter, path);
            }
        }
    }

    void ReadProfiles() {
        if (m_session == XR_NULL_HANDLE) {
            return;
        }
        for (uint32_t side = 0; side < 2; ++side) {
            XrInteractionProfileState state{XR_TYPE_INTERACTION_PROFILE_STATE};
            m_profiles[side].clear();
            if (XR_SUCCEEDED(
                    xrGetCurrentInteractionProfile(m_session, m_hands[side].path, &state)) &&
                state.interactionProfile != XR_NULL_PATH) {
                char text[XR_MAX_PATH_LENGTH];
                uint32_t length = 0;
                if (XR_SUCCEEDED(xrPathToString(m_instance, state.interactionProfile,
                                                XR_MAX_PATH_LENGTH, &length, text))) {
                    m_profiles[side] = text;
                }
            }
        }
    }

    void PollEvents() {
        XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
        while (xrPollEvent(m_instance, &event) == XR_SUCCESS) {
            switch (event.type) {
            case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
                ChangeState(reinterpret_cast<const XrEventDataSessionStateChanged&>(event).state);
                break;
            case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
                m_recentered = true;
                break;
            case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
                ReadProfiles();
                break;
            case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
                m_exit = true;
                break;
            case XR_TYPE_EVENT_DATA_DISPLAY_REFRESH_RATE_CHANGED_FB:
                m_rate = reinterpret_cast<const XrEventDataDisplayRefreshRateChangedFB&>(event)
                             .toDisplayRefreshRate;
                break;
            default:
                break;
            }
            event = {XR_TYPE_EVENT_DATA_BUFFER};
        }
    }

    void ChangeState(XrSessionState state) {
        m_state = state;
        if (state == XR_SESSION_STATE_READY) {
            XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO};
            begin.primaryViewConfigurationType = kStereo;
            Check(xrBeginSession(m_session, &begin), "xrBeginSession");
            m_running = true;
        } else if (state == XR_SESSION_STATE_STOPPING) {
            Check(xrEndSession(m_session), "xrEndSession");
            m_running = false;
        } else if (state == XR_SESSION_STATE_EXITING || state == XR_SESSION_STATE_LOSS_PENDING) {
            m_running = false;
            m_exit = true;
        }
    }

    void LocateViews(Frame& frame) {
        XrViewLocateInfo locate{XR_TYPE_VIEW_LOCATE_INFO};
        locate.viewConfigurationType = kStereo;
        locate.displayTime = m_displayTime;
        locate.space = m_space;
        XrViewState state{XR_TYPE_VIEW_STATE};
        uint32_t count = 0;
        m_views[0] = {XR_TYPE_VIEW};
        m_views[1] = {XR_TYPE_VIEW};
        Check(xrLocateViews(m_session, &locate, &state, 2, &count, m_views), "xrLocateViews");
        constexpr XrViewStateFlags kValid =
            XR_VIEW_STATE_POSITION_VALID_BIT | XR_VIEW_STATE_ORIENTATION_VALID_BIT;
        m_viewsValid = (state.viewStateFlags & kValid) == kValid;
        for (uint32_t eye = 0; eye < 2; ++eye) {
            const XrFovf& fov = m_views[eye].fov;
            frame.views[eye].pose = StagePose(m_views[eye].pose, m_viewsValid);
            frame.views[eye].fov = {fov.angleLeft, fov.angleRight, fov.angleUp, fov.angleDown};
        }
        frame.head = Locate(m_viewSpace);
    }

    Pose Locate(XrSpace space) const {
        XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
        Check(xrLocateSpace(space, m_space, m_displayTime, &location), "xrLocateSpace");
        return StagePose(location);
    }

    XrActionStateGetInfo StateInfo(Slot slot, XrPath hand) const {
        XrActionStateGetInfo get{XR_TYPE_ACTION_STATE_GET_INFO};
        get.action = m_actions[uint32_t(slot)];
        get.subactionPath = hand;
        return get;
    }

    float ReadFloat(Slot slot, XrPath hand) const {
        const XrActionStateGetInfo get = StateInfo(slot, hand);
        XrActionStateFloat state{XR_TYPE_ACTION_STATE_FLOAT};
        Check(xrGetActionStateFloat(m_session, &get, &state), "xrGetActionStateFloat");
        return state.isActive == XR_TRUE ? state.currentState : 0.0f;
    }

    bool ReadBool(Slot slot, XrPath hand) const {
        const XrActionStateGetInfo get = StateInfo(slot, hand);
        XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
        Check(xrGetActionStateBoolean(m_session, &get, &state), "xrGetActionStateBoolean");
        return state.isActive == XR_TRUE && state.currentState == XR_TRUE;
    }

    bool PoseActive(XrAction action, XrPath hand) const {
        XrActionStateGetInfo get{XR_TYPE_ACTION_STATE_GET_INFO};
        get.action = action;
        get.subactionPath = hand;
        XrActionStatePose state{XR_TYPE_ACTION_STATE_POSE};
        Check(xrGetActionStatePose(m_session, &get, &state), "xrGetActionStatePose");
        return state.isActive == XR_TRUE;
    }

    void ReadInputs(Frame& frame) {
        ERNIE_ZONE("XR/Actions");
        XrActiveActionSet active{m_actionSet, XR_NULL_PATH};
        XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO};
        sync.countActiveActionSets = 1;
        sync.activeActionSets = &active;
        Check(xrSyncActions(m_session, &sync), "xrSyncActions");
        for (uint32_t side = 0; side < 2; ++side) {
            const HandSpaces& hand = m_hands[side];
            Controller& controller = frame.controllers[side];
            controller.active = PoseActive(m_actions[uint32_t(Slot::GripPose)], hand.path);
            if (controller.active) {
                controller.pose = Locate(hand.grip);
                controller.aim = Locate(hand.aim);
            }
            controller.trigger = ReadFloat(Slot::Trigger, hand.path);
            controller.grip = ReadFloat(Slot::Squeeze, hand.path);
            const XrActionStateGetInfo get = StateInfo(Slot::Stick, hand.path);
            XrActionStateVector2f stick{XR_TYPE_ACTION_STATE_VECTOR2F};
            Check(xrGetActionStateVector2f(m_session, &get, &stick), "xrGetActionStateVector2f");
            if (stick.isActive == XR_TRUE) {
                controller.thumbstick = {stick.currentState.x, stick.currentState.y};
            }
            controller.buttons.primary = ReadBool(Slot::Primary, hand.path);
            controller.buttons.secondary = ReadBool(Slot::Secondary, hand.path);
            controller.buttons.thumbstick = ReadBool(Slot::StickClick, hand.path);
            controller.buttons.menu = ReadBool(Slot::Menu, hand.path);
        }
        LocateHands(frame);
        frame.profiles = {m_profiles[0], m_profiles[1]};
        frame.recentered = std::exchange(m_recentered, false);
        if (Has(Feature::EyeGaze) && PoseActive(m_gaze, XR_NULL_PATH)) {
            const Pose gaze = Locate(m_gazeSpace);
            if (gaze.valid) {
                frame.gaze = gaze;
            }
        }
    }

    void SendHaptics(const Frame& frame) {
        for (const HapticPulse& pulse : frame.haptics) {
            XrHapticVibration vibration{XR_TYPE_HAPTIC_VIBRATION};
            vibration.duration =
                pulse.duration > 0 ? XrDuration(pulse.duration * 1e9) : XR_MIN_HAPTIC_DURATION;
            vibration.frequency = pulse.frequency > 0 ? pulse.frequency : XR_FREQUENCY_UNSPECIFIED;
            vibration.amplitude = std::clamp(pulse.amplitude, 0.0f, 1.0f);
            XrHapticActionInfo info{XR_TYPE_HAPTIC_ACTION_INFO};
            info.action = m_actions[uint32_t(Slot::Haptic)];
            info.subactionPath = m_hands[uint32_t(pulse.hand)].path;
            Check(xrApplyHapticFeedback(m_session, &info,
                                        reinterpret_cast<XrHapticBaseHeader*>(&vibration)),
                  "xrApplyHapticFeedback");
        }
    }

    SessionDesc m_desc;
    XrInstance m_instance = XR_NULL_HANDLE;
    XrSystemId m_system = XR_NULL_SYSTEM_ID;
    std::vector<std::string> m_enabled;
    uint32_t m_features = 0;
    std::string m_runtimeName;
    std::string m_systemName;
    Uint2 m_recommended;
    Uint2 m_swapchainSize;

    PFN_xrGetVulkanGraphicsRequirements2KHR m_graphicsRequirements = nullptr;
    PFN_xrCreateVulkanInstanceKHR m_createVulkanInstance = nullptr;
    PFN_xrCreateVulkanDeviceKHR m_createVulkanDevice = nullptr;
    PFN_xrGetVulkanGraphicsDevice2KHR m_graphicsDevice = nullptr;
    PFN_xrEnumerateDisplayRefreshRatesFB m_enumerateRefreshRates = nullptr;
    PFN_xrGetDisplayRefreshRateFB m_getRefreshRate = nullptr;
    PFN_xrRequestDisplayRefreshRateFB m_requestRefreshRate = nullptr;
    PFN_xrCreateHandTrackerEXT m_createHandTracker = nullptr;
    PFN_xrDestroyHandTrackerEXT m_destroyHandTracker = nullptr;
    PFN_xrLocateHandJointsEXT m_locateHandJoints = nullptr;
    PFN_xrEnumeratePerformanceMetricsCounterPathsMETA m_enumerateCounters = nullptr;
    PFN_xrSetPerformanceMetricsStateMETA m_setMetricsState = nullptr;
    PFN_xrQueryPerformanceMetricsCounterMETA m_queryCounter = nullptr;
    bool m_graphicsChecked = false;
    bool m_hasTimespec = false;
#ifndef _WIN32
    PFN_xrConvertTimeToTimespecTimeKHR m_convertTime = nullptr;
#else
    void* m_convertTime = nullptr;
#endif

    XrActionSet m_actionSet = XR_NULL_HANDLE;
    std::array<XrAction, 10> m_actions{};
    XrAction m_gaze = XR_NULL_HANDLE;
    HandSpaces m_hands[2];

    Ref<device::Device> m_device;
    VkDevice m_vkDevice = VK_NULL_HANDLE;
    DeviceFunctions m_vk;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::array<Recording, kCommandRing> m_recordings;
    uint32_t m_nextRecording = 0;
    XrSession m_session = XR_NULL_HANDLE;
    XrSpace m_space = XR_NULL_HANDLE;
    XrSpace m_viewSpace = XR_NULL_HANDLE;
    XrSpace m_gazeSpace = XR_NULL_HANDLE;
    EyeSwapchain m_swapchains[2];
    EyeSwapchain m_depthSwapchains[2];
    bool m_depthLayer = false;
    XrEnvironmentBlendMode m_blendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    BlendMode m_blend = BlendMode::Opaque;
    XrHandTrackerEXT m_handTrackers[2] = {XR_NULL_HANDLE, XR_NULL_HANDLE};
    std::vector<std::pair<Counter, XrPath>> m_counters;
    Float2 m_playArea;
    std::string m_profiles[2];
    bool m_recentered = false;
    XrSessionState m_state = XR_SESSION_STATE_UNKNOWN;
    bool m_running = false;
    bool m_exit = false;
    bool m_frameBegun = false;
    XrTime m_displayTime = 0;
    XrTime m_lastDisplayTime = 0;
    XrView m_views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
    bool m_viewsValid = false;
    float m_rate = 0;
    bool m_hasRefreshRate = false;
    std::vector<float> m_rates;
};

}

std::unique_ptr<Runtime> CreateOpenXrRuntime(const SessionDesc& desc) {
    return std::make_unique<OpenXrRuntime>(desc);
}

}

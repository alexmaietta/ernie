// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/xr/profiles.h"

#include <algorithm>
#include <utility>

namespace ernie::xr {
namespace {

using Components = std::vector<std::pair<Slot, std::string>>;

InteractionProfile Make(std::string path, std::string extension, const Components& both,
                        const Components& left = {}, const Components& right = {}) {
    InteractionProfile profile{std::move(path), std::move(extension), {}};
    for (uint32_t hand = 0; hand < 2; ++hand) {
        Components all = {
            {Slot::GripPose, "grip/pose"}, {Slot::AimPose, "aim/pose"}, {Slot::Haptic, "haptic"}};
        all.insert(all.end(), both.begin(), both.end());
        const Components& own = hand == 0 ? left : right;
        all.insert(all.end(), own.begin(), own.end());
        for (const auto& [slot, component] : all) {
            profile.bindings.push_back({slot, hand, component});
        }
    }
    return profile;
}

InteractionProfile Touch(std::string path, std::string extension) {
    return Make(
        std::move(path), std::move(extension),
        {{Slot::Trigger, "trigger/value"},
         {Slot::Squeeze, "squeeze/value"},
         {Slot::Stick, "thumbstick"},
         {Slot::StickClick, "thumbstick/click"}},
        {{Slot::Primary, "x/click"}, {Slot::Secondary, "y/click"}, {Slot::Menu, "menu/click"}},
        {{Slot::Primary, "a/click"}, {Slot::Secondary, "b/click"}});
}

std::vector<InteractionProfile> MakeProfiles() {
    return {
        Touch("/interaction_profiles/oculus/touch_controller", ""),
        Touch("/interaction_profiles/meta/touch_plus_controller", "XR_META_touch_controller_plus"),
        Touch("/interaction_profiles/facebook/touch_controller_pro", "XR_FB_touch_controller_pro"),
        Touch("/interaction_profiles/bytedance/pico4_controller", "XR_BD_controller_interaction"),
        Make("/interaction_profiles/valve/index_controller", "",
             {{Slot::Trigger, "trigger/value"},
              {Slot::Squeeze, "squeeze/value"},
              {Slot::Stick, "thumbstick"},
              {Slot::StickClick, "thumbstick/click"},
              {Slot::Primary, "a/click"},
              {Slot::Secondary, "b/click"}}),
        Make("/interaction_profiles/htc/vive_controller", "",
             {{Slot::Trigger, "trigger/value"},
              {Slot::Squeeze, "squeeze/click"},
              {Slot::Stick, "trackpad"},
              {Slot::StickClick, "trackpad/click"},
              {Slot::Primary, "trackpad/click"},
              {Slot::Menu, "menu/click"}}),
        Make("/interaction_profiles/microsoft/motion_controller", "",
             {{Slot::Trigger, "trigger/value"},
              {Slot::Squeeze, "squeeze/click"},
              {Slot::Stick, "thumbstick"},
              {Slot::StickClick, "thumbstick/click"},
              {Slot::Primary, "trackpad/click"},
              {Slot::Menu, "menu/click"}}),
        Make("/interaction_profiles/khr/simple_controller", "",
             {{Slot::Trigger, "select/click"}, {Slot::Menu, "menu/click"}}),
    };
}

float Value(const std::map<std::string, float, std::less<>>& pressed,
            const std::string& component) {
    const auto found = pressed.find(component);
    return found != pressed.end() ? found->second : 0.0f;
}

}

const std::vector<InteractionProfile>& InteractionProfiles() {
    static const std::vector<InteractionProfile> profiles = MakeProfiles();
    return profiles;
}

const InteractionProfile* FindProfile(std::string_view path) {
    for (const InteractionProfile& profile : InteractionProfiles()) {
        if (profile.path == path) {
            return &profile;
        }
    }
    return nullptr;
}

bool HasComponent(const InteractionProfile& profile, uint32_t hand, std::string_view component) {
    return std::ranges::any_of(profile.bindings, [&](const ComponentBinding& binding) {
        if (binding.hand != hand || binding.slot == Slot::GripPose ||
            binding.slot == Slot::AimPose || binding.slot == Slot::Haptic) {
            return false;
        }
        return binding.component == component ||
               (binding.slot == Slot::Stick &&
                (component == binding.component + "/x" || component == binding.component + "/y"));
    });
}

void ApplyComponents(const InteractionProfile& profile, uint32_t hand,
                     const std::map<std::string, float, std::less<>>& pressed,
                     Controller& controller) {
    for (const ComponentBinding& binding : profile.bindings) {
        if (binding.hand != hand) {
            continue;
        }
        const float value = Value(pressed, binding.component);
        const bool down = value >= 0.5f;
        switch (binding.slot) {
        case Slot::Trigger:
            controller.trigger = std::max(controller.trigger, value);
            break;
        case Slot::Squeeze:
            controller.grip = std::max(controller.grip, value);
            break;
        case Slot::Stick: {
            const float x = Value(pressed, binding.component + "/x");
            const float y = Value(pressed, binding.component + "/y");
            if (x != 0 || y != 0) {
                controller.thumbstick = {x, y};
            }
            break;
        }
        case Slot::StickClick:
            controller.buttons.thumbstick = controller.buttons.thumbstick || down;
            break;
        case Slot::Primary:
            controller.buttons.primary = controller.buttons.primary || down;
            break;
        case Slot::Secondary:
            controller.buttons.secondary = controller.buttons.secondary || down;
            break;
        case Slot::Menu:
            controller.buttons.menu = controller.buttons.menu || down;
            break;
        default:
            break;
        }
    }
}

}

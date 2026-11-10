// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/xr/frame.h"

namespace ernie::xr {

enum class Slot : uint32_t {
    GripPose,
    AimPose,
    Trigger,
    Squeeze,
    Stick,
    StickClick,
    Primary,
    Secondary,
    Menu,
    Haptic,
};

struct ComponentBinding {
    Slot slot;
    uint32_t hand;
    std::string component;
};

struct InteractionProfile {
    std::string path;
    std::string extension;
    std::vector<ComponentBinding> bindings;
};

const std::vector<InteractionProfile>& InteractionProfiles();

const InteractionProfile* FindProfile(std::string_view path);

bool HasComponent(const InteractionProfile& profile, uint32_t hand, std::string_view component);

void ApplyComponents(const InteractionProfile& profile, uint32_t hand,
                     const std::map<std::string, float, std::less<>>& pressed,
                     Controller& controller);

}

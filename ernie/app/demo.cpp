// SPDX-License-Identifier: BSD-3-Clause
/// @file demo.cpp
/// @brief Stepping demos, and the registry of them.
#include "ernie/app/demo.h"

#include "ernie/util/text.h"

#include <algorithm>
#include <tuple>

namespace ernie::app {
namespace {

std::vector<DemoInfo>& Registry() {
    static std::vector<DemoInfo> demos;
    return demos;
}

/// @return A demo name compared loosely: lower case, `_` as a space.
std::string Normalized(std::string_view name) {
    std::string text = ToLower(name);
    std::ranges::replace(text, '_', ' ');
    return text;
}

}  // namespace

void Demo::Advance() {
    Step();
    ++frame_;
    time_ += fixedDt;
}

std::string DemoInfo::Label() const {
    return category + ": " + name;
}

std::string DemoInfo::Id() const {
    std::string id = "demo_" + category + "_" + name;
    std::ranges::replace(id, ' ', '_');
    return id;
}

std::vector<DemoInfo> RegisteredDemos() {
    std::vector<DemoInfo> demos = Registry();
    std::ranges::sort(demos, {},
                      [](const DemoInfo& demo) { return std::tie(demo.category, demo.name); });
    return demos;
}

const DemoInfo* FindDemo(const std::vector<DemoInfo>& demos, std::string_view name) {
    const std::string wanted = Normalized(name);
    const auto found = std::ranges::find_if(demos, [&](const DemoInfo& demo) {
        return Normalized(demo.Id()) == wanted || Normalized(demo.Label()) == wanted ||
               Normalized(demo.category + "/" + demo.name) == wanted;
    });
    return found != demos.end() ? &*found : nullptr;
}

DemoRegistration::DemoRegistration(DemoInfo info) {
    Registry().push_back(std::move(info));
}

}  // namespace ernie::app

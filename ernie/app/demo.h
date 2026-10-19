// SPDX-License-Identifier: BSD-3-Clause
/// @file demo.h
/// @brief Demos: a base class building a scene and stepping it, and the registry the Demos panel
/// lists them from.
///
///     class Bounce final : public Demo {
///       public:
///         explicit Bounce(Scene& scene) : Demo(scene) { Sphere::Define(scene, "/World/Ball"); }
///         void Step() override { ... }
///     };
///     ERNIE_REGISTER_DEMO(Bounce, "basics", "bounce", "A ball bouncing on the ground.");
///
/// References:
/// - Newton's examples (`newton.examples`: `Example` and the example browser),
///   https://github.com/newton-physics/newton
#pragma once

#include "ernie/scene/scene.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::app {

/// @brief A demo: a scene built in the constructor and advanced by `Step`, a fixed time step at a
/// time. The viewer's Demos panel runs, pauses, steps and resets it.
class Demo {
  public:
    /// @param scene The scene the demo builds into, empty to begin with.
    explicit Demo(Scene& scene) : scene_(scene) {}
    virtual ~Demo() = default;
    Demo(const Demo&) = delete;
    Demo& operator=(const Demo&) = delete;

    /// @brief Advances the scene by `fixedDt`; nothing by default.
    virtual void Step() {}
    /// @brief Draws the demo's own options in the Demos panel; nothing by default.
    virtual void Gui() {}
    /// @brief Checks after each step of a test run; throw to fail.
    virtual void TestPostStep() {}
    /// @brief Checks at the end of a test run; throw to fail.
    virtual void TestFinal() {}

    /// @brief Steps once: `Step`, then the frame and time move on.
    void Advance();

    std::uint64_t Frame() const {
        return frame_;
    }
    /// @return Seconds simulated.
    double Time() const {
        return time_;
    }

    /// Seconds a step simulates.
    double fixedDt = 1.0 / 60.0;
    /// What the demo shows, for the Demos panel.
    std::string description;

  protected:
    Scene& GetScene() {
        return scene_;
    }

  private:
    Scene& scene_;
    std::uint64_t frame_ = 0;
    double time_ = 0.0;
};

/// @brief A registered demo: where the Demos panel lists it and how to make it.
struct DemoInfo {
    std::string category;
    std::string name;
    std::string description;
    std::function<std::unique_ptr<Demo>(Scene&)> create;

    /// @return "category: name", as the panel shows it.
    std::string Label() const;
    /// @return "demo_<category>_<name>", spaces as underscores, as its files are named.
    std::string Id() const;
};

/// @return Every registered demo, by category, then name.
std::vector<DemoInfo> RegisteredDemos();

/// @return The demo an id, "category/name" or a label names, `_` and spaces alike; null for none.
const DemoInfo* FindDemo(const std::vector<DemoInfo>& demos, std::string_view name);

/// @brief Registers a demo at static initialization; `ERNIE_REGISTER_DEMO` makes one.
struct DemoRegistration {
    explicit DemoRegistration(DemoInfo info);
};

}  // namespace ernie::app

/// Registers `Class`, constructed from a `Scene&`, under a category and name.
#define ERNIE_REGISTER_DEMO(Class, category, name, text)                                           \
    const ::ernie::app::DemoRegistration ernieDemo##Class(                                         \
        {category, name, text,                                                                     \
         [](::ernie::Scene& scene) { return std::make_unique<Class>(scene); }})

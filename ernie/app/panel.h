// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ernie::app {

class Viewer;

/// Where a panel docks in the viewer's default layout, around the
/// viewport.
enum class Dock {
    Left,
    Right,  ///< The right column's upper part (the whole column if
            ///< nothing docks below).
    Bottom,
    Floating,    ///< Not docked: a free window.
    RightLower,  ///< The right column's lower part, below `Right`.
};

/// A dockable window of the viewer. Subclasses draw with Dear ImGui
/// (`#include <imgui.h>`) in `Draw`, between the window's `Begin` and
/// `End`, which the viewer calls.
class Panel {
  public:
    explicit Panel(std::string title, Dock dock = Dock::Right)
        : title_(std::move(title)), dock_(dock) {}
    virtual ~Panel() = default;
    Panel(const Panel&) = delete;
    Panel& operator=(const Panel&) = delete;

    /// Title, also the panel's id in saved layouts.
    const std::string& Title() const {
        return title_;
    }

    /// Where the panel docks in the default layout; a saved layout
    /// overrides it.
    Dock GetDock() const {
        return dock_;
    }

    bool IsVisible() const {
        return visible_;
    }

    void SetVisible(bool visible) {
        visible_ = visible;
    }

    /// Draws the panel's contents for the current frame.
    virtual void Draw(Viewer& viewer) = 0;

  private:
    std::string title_;
    Dock dock_;
    bool visible_ = true;
};

/// A panel that draws its contents by calling a function.
class FunctionPanel final : public Panel {
  public:
    FunctionPanel(std::string title, Dock dock,
                  std::function<void(Viewer&)> draw)
        : Panel(std::move(title), dock), draw_(std::move(draw)) {}

    void Draw(Viewer& viewer) override {
        if (draw_) {
            draw_(viewer);
        }
    }

  private:
    std::function<void(Viewer&)> draw_;
};

/// A panel of simple widgets described by data rather than ImGui
/// calls, so code without ImGui bindings (Python) can build one. Each
/// widget calls its callback when the user changes it; `Set` changes
/// a widget's value from code without calling it.
class WidgetPanel final : public Panel {
  public:
    using Id = std::uint32_t;

    explicit WidgetPanel(std::string title, Dock dock = Dock::Right);
    ~WidgetPanel() override;

    Id Text(std::string text);
    Id Separator();
    Id Button(std::string label, std::function<void()> onClick);
    Id Checkbox(std::string label, bool value,
                std::function<void(bool)> onChange);
    Id Slider(std::string label, double min, double max, double value,
              std::function<void(double)> onChange);
    Id Combo(std::string label, std::vector<std::string> items,
             std::size_t selected,
             std::function<void(std::size_t)> onChange);

    /// A field of editable text; `onChange` gets it after each edit.
    Id InputText(std::string label, std::string value,
                 std::function<void(const std::string&)> onChange);
    /// An RGB colour swatch with a picker, components in [0, 1].
    Id Color(std::string label, std::array<double, 3> value,
             std::function<void(std::array<double, 3>)> onChange);

    /// A line plot of the values `source` returns, read each frame.
    Id Plot(std::string label,
            std::function<std::vector<float>()> source);

    /// Sets the value of a checkbox (`bool`), slider (`double`),
    /// combo (index) or colour, or the text of a text or text field.
    void Set(Id widget, bool value);
    void Set(Id widget, double value);
    void Set(Id widget, std::size_t value);
    void Set(Id widget, std::string value);
    void Set(Id widget, std::array<double, 3> value);

    void Remove(Id widget);
    void Clear();

    void Draw(Viewer& viewer) override;

  private:
    struct Widget;
    std::vector<std::unique_ptr<Widget>> widgets_;
    Id next_ = 1;
};

/// ImGui::InputText on a std::string, which grows as the user types;
/// `flags` are ImGuiInputTextFlags.
bool InputString(const char* label, std::string& text, int flags = 0);

}  // namespace ernie::app

#pragma once

/// Ernie's GUI viewer in one include, with Dear ImGui and ImPlot for drawing panels:
///
///     #include "ernie/app/app.h"
///
///     class SimPanel final : public ernie::app::Panel {
///       public:
///         SimPanel() : Panel("Sim", ernie::app::Dock::Left) {}
///         void Draw(ernie::app::Viewer& viewer) override {
///             ImGui::SliderFloat("Speed", &speed_, 0.0f, 10.0f);
///         }
///       private:
///         float speed_ = 1.0f;
///     };
///
///     ernie::Scene scene;
///     ernie::app::Viewer viewer(scene);
///     viewer.AddPanel<SimPanel>();
///     viewer.OnPreRender([](ernie::app::Viewer& v, const ernie::app::FrameInfo& frame) { ... });
///     viewer.Run();

#include "ernie/app/icons.h"
#include "ernie/app/panel.h"
#include "ernie/app/theme.h"
#include "ernie/app/viewer.h"

#include <imgui.h>
#include <implot.h>

#pragma once

#include "ernie/app/panel.h"

#include <memory>

namespace ernie::app {

/// The Stage panel: the scene's prim tree, with selection, framing, visibility toggles and a
/// filter; the session's own prims show on request.
std::unique_ptr<Panel> MakeStagePanel();

/// The Properties panel: the selected prim's attributes as widgets by value type, editing through
/// the viewer's undo history, with a reset back to each attribute's fallback; bulk data shows its
/// shape.
std::unique_ptr<Panel> MakePropertiesPanel();

/// The Settings panel. Render: the viewport's RenderSettings prim and the settings the renderer
/// reads (image, culling and level of detail, shadows), with their fallbacks. Viewer: the
/// viewport's own (renderer, shown var, lighting, wireframe, vsync, gizmo, theme).
std::unique_ptr<Panel> MakeSettingsPanel();

/// The Camera panel: the active camera's lens, exposure and sensor, its channels' quantum
/// efficiency over wavelength and as binned into its bands, and Create Camera from View.
std::unique_ptr<Panel> MakeCameraPanel();

/// The Log panel: the viewer's messages (`Viewer::Log`), filtered by level and text.
std::unique_ptr<Panel> MakeLogPanel();

}  // namespace ernie::app

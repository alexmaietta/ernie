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

}  // namespace ernie::app

// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/app/panel.h"

#include <memory>

namespace ernie::app {

/// The Stage panel: the scene's prim tree, with selection, framing,
/// visibility toggles and a filter; the session's own prims show on
/// request.
std::unique_ptr<Panel> MakeStagePanel();

/// The Properties panel: the selected prim's attributes as widgets by
/// value type, editing through the viewer's undo history, with a
/// reset back to each attribute's fallback; bulk data shows its
/// shape.
std::unique_ptr<Panel> MakePropertiesPanel();

/// The Settings panel. Render: the viewport's RenderSettings prim and
/// the settings the renderer reads (image, culling and level of
/// detail, shadows), with their fallbacks. Viewer: the viewport's own
/// (renderer, shown var, lighting, wireframe, vsync, gizmo, theme).
std::unique_ptr<Panel> MakeSettingsPanel();

/// The Camera panel: the active camera's lens, exposure and sensor,
/// its channels' quantum efficiency over wavelength and as binned
/// into its bands, and Create Camera from View.
std::unique_ptr<Panel> MakeCameraPanel();

/// The Log panel: the viewer's messages (`Viewer::Log`), filtered by
/// level and text.
std::unique_ptr<Panel> MakeLogPanel();

/// The Profiler panel, hidden until shown: Ernie's profiler
/// (ernie/util/profiler.h) recording while it shows, with frame
/// times, a timeline of the CPU and GPU zones of a frame, and their
/// totals, after Omniverse Kit's profiler window.
std::unique_ptr<Panel> MakeProfilerPanel();

/// The Timeline panel, a strip under the viewport after Kit's
/// timeline: the stage's start time code, a ruler to scrub across its
/// range, its end, the current time code, the transport (first frame,
/// a frame back, play or pause, stop, a frame on, last frame), loop,
/// and a menu of the frame rate and playing every frame.
std::unique_ptr<Panel> MakeTimelinePanel();

/// The Capture panel: where captures go and how they are named, a
/// screenshot (F10), recording a sequence of the timeline's frames
/// (optionally into a video with ffmpeg), and the recent captures.
std::unique_ptr<Panel> MakeCapturePanel();

}  // namespace ernie::app

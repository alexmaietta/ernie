// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <functional>
#include <memory>
#include <string>

#include "ernie/device/device.h"
#include "ernie/scene/scene.h"
#include "ernie/viewer/shell/capture.h"
#include "ernie/viewer/shell/editor.h"
#include "ernie/viewer/shell/selection.h"
#include "ernie/viewer/theme/theme.h"
#include "ernie/viewer/viewer.h"
#include "ernie/viewer/widgets/file_dialog.h"

namespace ernie::app {

struct Panel {
    std::string title;
    Dock dock;
    std::function<void()> draw;
    bool visible = true;
    std::function<void()> hidden;
    bool builtin = false;
};

Panel CreateInfoPanel(Ref<Scene> scene, Ref<device::Device> device,
                      std::shared_ptr<RenderDelegate> delegate);

Panel CreateLogPanel(const Theme& theme);

Panel CreateCapturePanel(Capture& capture);

Panel CreateProfilerPanel();

Panel CreateMemoryPanel(Ref<device::Device> device);

Panel CreateStagePanel(Ref<Scene> scene, Selection& selection, Editor& editor, const Theme& theme);

Panel CreatePropertiesPanel(Ref<Scene> scene, Selection& selection, Editor& editor,
                            FileDialog& dialog, const Theme& theme);

void DrawControls(bool& open, const Theme& theme);

void DrawAbout(bool& open, device::Device& device);

}

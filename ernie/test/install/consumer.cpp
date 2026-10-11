// SPDX-License-Identifier: BSD-3-Clause
// Uses the installed headers and libernie the way an application
// does: a scene, the render backends, a Renderer when a GPU is there,
// and a viewer panel drawn with Dear ImGui when the app is installed.
// Building it checks that the headers are complete and the symbols
// exported.

#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/renderer.h"

#if __has_include("ernie/app/panel.h")
#include "ernie/app/panel.h"

#include <imgui.h>
#endif

#include <cstdio>
#include <exception>
#include <memory>

namespace {

#if __has_include("ernie/app/panel.h")
class HelloPanel final : public ernie::app::Panel {
  public:
    HelloPanel() : Panel("Hello") {}
    void Draw(ernie::app::Viewer&) override {
        ImGui::TextUnformatted("Hello from an installed Ernie");
    }
};
#endif

}  // namespace

int main() {
    ernie::Scene scene;
    scene.Define("/World/Cube", ernie::tokens::kCube)
        .Set(ernie::tokens::kSize, 2.0);
    if (!scene.Has("/World/Cube")) {
        std::fprintf(stderr, "the scene lost its cube\n");
        return 1;
    }

    if (ernie::RenderBackends().empty()) {
        std::fprintf(stderr, "no render backends\n");
        return 1;
    }

#if __has_include("ernie/app/panel.h")
    const std::unique_ptr<ernie::app::Panel> panel =
        std::make_unique<HelloPanel>();
    std::printf("panel: %s\n", panel->Title().c_str());
#endif

    // Without a GPU (or a Vulkan driver) there is nothing more to
    // check here.
    try {
        const ernie::Renderer renderer;
        std::printf("renderer: %s on %s\n",
                    renderer.Info().name.c_str(),
                    renderer.DeviceName().c_str());
    } catch (const std::exception& error) {
        std::printf("no renderer: %s\n", error.what());
    }
    return 0;
}

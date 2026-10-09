// The `ernie` executable: opens the viewer on a small default scene, a ground plane with a few
// primitives under a sun, until the window closes.

#include "ernie/app/viewer.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <exception>
#include <string_view>

namespace {

constexpr const char* kUsage = R"(Usage: ernie [options]

Opens Ernie's interactive viewer on a default scene.

Options:
  --renderer NAME   Render backend to start with (default: raster)
  --validation      Enable the Vulkan validation layer when installed
  --vsync           Wait for the display's refresh (default: draw as fast as possible)
  --version         Show Ernie's version and exit
  -h, --help        Show this help and exit
)";

/// A ground plane with a cube, a sphere and a cylinder on it, lit by a sun.
ernie::Scene DefaultScene() {
    using namespace ernie;
    Scene scene;
    scene.Define("/World/Ground", tokens::kGroundPlane);
    scene.Define("/World/Cube", tokens::kCube)
        .Set(tokens::kSize, 2.0)
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(-3.0, 0.0, 1.0)))
        .Set(tokens::kDisplayColor, glm::vec3(0.8f, 0.3f, 0.2f));
    scene.Define("/World/Sphere", tokens::kSphere)
        .Set(tokens::kGeomRadius, 1.0)
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 1.0)))
        .Set(tokens::kDisplayColor, glm::vec3(0.2f, 0.5f, 0.8f));
    scene.Define("/World/Cylinder", tokens::kCylinder)
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(3.0, 0.0, 1.0)))
        .Set(tokens::kDisplayColor, glm::vec3(0.3f, 0.7f, 0.3f));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, DistantLightTransform({0.4, -0.6, 0.7}));
    return scene;
}

}  // namespace

int main(int argc, char** argv) {
    ernie::app::ViewerOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::fputs(kUsage, stdout);
            return 0;
        }
        if (arg == "--version") {
            std::puts("ernie " ERNIE_VERSION);
            return 0;
        }
        if (arg == "--validation") {
            options.validation = true;
        } else if (arg == "--vsync") {
            options.vsync = true;
        } else if (arg == "--renderer" && i + 1 < argc) {
            options.renderer = argv[++i];
        } else {
            std::fprintf(stderr, "ernie: unknown argument '%s'\n\n%s", argv[i], kUsage);
            return 2;
        }
    }
    try {
        ernie::Scene scene = DefaultScene();
        ernie::app::Viewer viewer(scene, options);
        viewer.Run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie: %s\n", error.what());
        return 1;
    }
    return 0;
}

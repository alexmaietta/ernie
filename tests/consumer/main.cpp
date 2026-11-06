// SPDX-License-Identifier: BSD-3-Clause
#include <cstdio>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/ernie.h"

namespace device = ernie::device;

int ListAdapters() {
    for (const device::AdapterInfo& adapter : device::EnumerateAdapters()) {
        std::printf(
            "%s: %s (%.*s)%s%s\n", adapter.alias.c_str(), adapter.name.c_str(),
            int(device::BackendName(adapter.backend).size()),
            device::BackendName(adapter.backend).data(),
            adapter.unsupported.empty() ? "" : ", unsupported: ", adapter.unsupported.c_str());
    }
    return 0;
}

int RoundTrip() {
    const ernie::Ref<ernie::Scene> scene = ernie::Scene::Create();
    std::printf("using %s\n", scene->GetDevice().Adapter().name.c_str());
    scene->Define<ernie::Xform>("/World");
    const std::vector<float> sent = {1, 2, 3, 5, 8, 13, 21, 34};
    for (size_t index = 0; index < sent.size(); ++index) {
        scene->Define<ernie::Sphere>("/World/Ball" + std::to_string(index));
    }
    ernie::Attribute<float> radius = scene->Select("/World/*").Get<float>("radius");
    radius.Set(sent);
    scene->Publish();
    if (radius.Get() != sent) {
        std::fprintf(stderr, "the radii read back differ from what was written\n");
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string_view(argv[1]) == "round-trip") {
            return RoundTrip();
        }
        return ListAdapters();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}

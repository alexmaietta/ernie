// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::app {

struct Selection {
    std::vector<std::string> paths;

    bool Contains(std::string_view path) const {
        return std::ranges::find(paths, path) != paths.end();
    }

    void Set(std::string path) {
        paths = {std::move(path)};
    }

    void Toggle(const std::string& path) {
        if (const auto found = std::ranges::find(paths, path); found != paths.end()) {
            paths.erase(found);
        } else {
            paths.push_back(path);
        }
    }

    void Clear() {
        paths.clear();
    }
};

}

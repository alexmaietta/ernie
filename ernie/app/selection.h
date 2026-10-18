// SPDX-License-Identifier: BSD-3-Clause
/// @file selection.h
/// @brief The prims selected in the viewer.
#pragma once

#include "ernie/scene/path.h"

#include <algorithm>
#include <vector>

namespace ernie::app {

/// @brief Selected prims in the order they were picked.
class Selection {
  public:
    const std::vector<Path>& Paths() const {
        return paths_;
    }
    bool Empty() const {
        return paths_.empty();
    }
    bool Contains(const Path& path) const {
        return std::ranges::find(paths_, path) != paths_.end();
    }
    /// @brief Selects `path` alone.
    void Set(const Path& path) {
        paths_ = {path};
    }
    /// @brief Adds `path`, or removes it when selected.
    void Toggle(const Path& path) {
        if (const auto found = std::ranges::find(paths_, path); found != paths_.end()) {
            paths_.erase(found);
        } else {
            paths_.push_back(path);
        }
    }
    void Clear() {
        paths_.clear();
    }

  private:
    std::vector<Path> paths_;
};

}  // namespace ernie::app

// SPDX-License-Identifier: BSD-3-Clause
#include "held.h"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ernie::python {

namespace {

struct Child {
    const void* child;
    std::function<bool()> alive;
};

/// What each owner holds; slots that expired and children let go are
/// pruned as they are met.
struct Entry {
    std::vector<std::weak_ptr<nb::object>> held;
    std::vector<Child> children;
};

std::unordered_map<const void*, Entry>& Registry() {
    static auto* registry =
        new std::unordered_map<const void*, Entry>;
    return *registry;
}

/// Calls `each` on the live objects `owner` and its children hold.
template <typename Each>
int ForEachHeld(const void* owner, const Each& each, int depth = 0) {
    auto found = Registry().find(owner);
    if (found == Registry().end() || depth > 4) {
        return 0;
    }
    Entry& entry = found->second;
    std::erase_if(entry.held,
                  [](const std::weak_ptr<nb::object>& slot) {
                      return slot.expired();
                  });
    std::erase_if(entry.children,
                  [](const Child& child) { return !child.alive(); });
    // Copied: visiting may not change the registry, but be safe.
    const auto held = entry.held;
    const auto children = entry.children;
    for (const std::weak_ptr<nb::object>& slot : held) {
        if (const Held object = slot.lock()) {
            if (const int result = each(*object)) {
                return result;
            }
        }
    }
    for (const Child& child : children) {
        if (const int result =
                ForEachHeld(child.child, each, depth + 1)) {
            return result;
        }
    }
    return 0;
}

}  // namespace

Held Hold(const void* owner, nb::object object) {
    Held held(new nb::object(std::move(object)),
              [](nb::object* slot) {
                  nb::gil_scoped_acquire gil;
                  delete slot;
              });
    std::vector<std::weak_ptr<nb::object>>& slots =
        Registry()[owner].held;
    std::erase_if(slots, [](const std::weak_ptr<nb::object>& slot) {
        return slot.expired();
    });
    slots.push_back(held);
    return held;
}

void Adopt(const void* owner, const void* child,
           std::function<bool()> alive) {
    // Once: a child visited twice would count its objects twice.
    std::vector<Child>& children = Registry()[owner].children;
    if (std::none_of(
            children.begin(), children.end(),
            [child](const Child& c) { return c.child == child; })) {
        children.push_back({child, std::move(alive)});
    }
}

int TraverseHeld(const void* owner, visitproc visit, void* arg) {
    return ForEachHeld(owner, [&](nb::object& object) -> int {
        if (object.is_valid()) {
            Py_VISIT(object.ptr());
        }
        return 0;
    });
}

void ClearHeld(const void* owner) {
    // Released after the walk: freeing may run code that holds more.
    std::vector<nb::object> released;
    ForEachHeld(owner, [&](nb::object& object) -> int {
        released.push_back(std::move(object));
        object = nb::object();
        return 0;
    });
}

}  // namespace ernie::python

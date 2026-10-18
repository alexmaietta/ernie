// SPDX-License-Identifier: BSD-3-Clause
/// @file blackboard.h
/// @brief A per-graph store where passes leave structs of handles for later passes to find by
/// type, rather than in renderer members.
///
///     struct Hits { graph::Image ids, depth; };
///     graph.Board().Put(Hits{ids, depth});
///     // Later, in another feature's setup:
///     const Hits& hits = graph.Board().Get<Hits>();
///
/// References:
/// - Google, Filament's FrameGraph Blackboard. https://github.com/google/filament
#pragma once

#include <any>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

namespace ernie::graph {

/// @brief One value of each type, such as a feature's handles.
class Blackboard {
  public:
    /// @brief Stores a value, replacing one of the same type.
    /// @return The stored value.
    template <typename T> T& Put(T value) {
        std::any& slot = values_[typeid(T)];
        slot = std::move(value);
        return std::any_cast<T&>(slot);
    }
    /// @return The value of type `T`, or null.
    template <typename T> T* Find() {
        const auto found = values_.find(typeid(T));
        return found == values_.end() ? nullptr : std::any_cast<T>(&found->second);
    }
    template <typename T> const T* Find() const {
        const auto found = values_.find(typeid(T));
        return found == values_.end() ? nullptr : std::any_cast<T>(&found->second);
    }
    /// @return The value of type `T`.
    /// @throws std::out_of_range if none was put.
    template <typename T> T& Get() {
        if (T* value = Find<T>()) {
            return *value;
        }
        throw std::out_of_range(std::string("The blackboard has no ") + typeid(T).name());
    }
    template <typename T> bool Has() const {
        return values_.contains(typeid(T));
    }
    void Clear() {
        values_.clear();
    }

  private:
    std::unordered_map<std::type_index, std::any> values_;
};

}  // namespace ernie::graph

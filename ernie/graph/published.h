// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace ernie::graph {

template <class T> class Published {
  public:
    explicit Published(T initial = T()) : m_value(std::make_shared<const T>(std::move(initial))) {}

    std::shared_ptr<const T> Get() const {
        const std::lock_guard lock(m_mutex);
        return m_value;
    }

    template <class Edit> auto Update(Edit&& edit) {
        const std::lock_guard lock(m_mutex);
        auto copy = std::make_shared<T>(*m_value);
        if constexpr (std::is_void_v<decltype(edit(*copy))>) {
            edit(*copy);
            m_value = std::move(copy);
        } else {
            auto result = edit(*copy);
            m_value = std::move(copy);
            return result;
        }
    }

  private:
    mutable std::mutex m_mutex;
    std::shared_ptr<const T> m_value;
};

}

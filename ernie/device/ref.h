// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace ernie {

class RefCounted {
  public:
    RefCounted(const RefCounted&) = delete;
    RefCounted& operator=(const RefCounted&) = delete;

    void AddRef() const noexcept {
        m_refs.fetch_add(1, std::memory_order_relaxed);
    }

    void Release() const noexcept {
        if (m_refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            const_cast<RefCounted*>(this)->LastReleased();
        }
    }

    uint32_t RefCount() const noexcept {
        return m_refs.load(std::memory_order_relaxed);
    }

  protected:
    RefCounted() = default;
    virtual ~RefCounted() = default;

    virtual void LastReleased() noexcept {
        delete this;
    }

  private:
    mutable std::atomic<uint32_t> m_refs{0};
};

template <class T> class Ref {
  public:
    Ref() noexcept = default;

    Ref(std::nullptr_t) noexcept {}

    explicit Ref(T* object) noexcept : m_object(object) {
        if (m_object) {
            m_object->AddRef();
        }
    }

    Ref(const Ref& other) noexcept : Ref(other.m_object) {}

    Ref(Ref&& other) noexcept : m_object(std::exchange(other.m_object, nullptr)) {}

    template <class U, class = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    Ref(const Ref<U>& other) noexcept : Ref(other.Get()) {}

    template <class U, class = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    Ref(Ref<U>&& other) noexcept : m_object(other.Detach()) {}

    ~Ref() {
        if (m_object) {
            m_object->Release();
        }
    }

    Ref& operator=(Ref other) noexcept {
        std::swap(m_object, other.m_object);
        return *this;
    }

    T* Get() const noexcept {
        return m_object;
    }

    T* operator->() const noexcept {
        return m_object;
    }

    T& operator*() const noexcept {
        return *m_object;
    }

    explicit operator bool() const noexcept {
        return m_object != nullptr;
    }

    [[nodiscard]] T* Detach() noexcept {
        return std::exchange(m_object, nullptr);
    }

    friend bool operator==(const Ref& a, const Ref& b) noexcept {
        return a.m_object == b.m_object;
    }

    friend bool operator==(const Ref& a, std::nullptr_t) noexcept {
        return a.m_object == nullptr;
    }

  private:
    T* m_object = nullptr;
};

template <class T, class... Args> Ref<T> MakeRef(Args&&... args) {
    return Ref<T>(new T(std::forward<Args>(args)...));
}

}

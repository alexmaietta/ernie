// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/ref.h"

#include <utility>

#include <gtest/gtest.h>

#include "ernie/device/stream.h"

namespace ernie {
namespace {

int g_alive = 0;

class Counted : public RefCounted {
  public:
    Counted() {
        ++g_alive;
    }

    ~Counted() override {
        --g_alive;
    }
};

class Derived : public Counted {};

TEST(RefTest, OwnsWhileAnyReferenceRemains) {
    {
        Ref<Counted> first(new Counted);
        EXPECT_EQ(g_alive, 1);
        Ref<Counted> copy = first;
        Ref<Counted> moved = std::move(first);
        EXPECT_FALSE(first);
        EXPECT_EQ(copy.Get(), moved.Get());
        copy = nullptr;
        EXPECT_EQ(g_alive, 1);
        moved = Ref<Counted>(new Counted);
        EXPECT_EQ(g_alive, 1);
    }
    EXPECT_EQ(g_alive, 0);
}

TEST(RefTest, ConvertsToItsBases) {
    Ref<Derived> derived(new Derived);
    Ref<Counted> base = derived;
    EXPECT_EQ(base.Get(), derived.Get());
    Ref<Counted> taken = std::move(derived);
    EXPECT_FALSE(derived);
    base = nullptr;
    taken = nullptr;
    EXPECT_EQ(g_alive, 0);
}

static_assert(Stream::None().IsNone());
static_assert(Stream::Host().IsHost());
static_assert(Stream::Default().IsCompute() && Stream::Default().IsDefault());
static_assert(!Stream::Compute(&g_alive).IsDefault());

}
}

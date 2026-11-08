// SPDX-License-Identifier: BSD-3-Clause
#include <string>

namespace {

long g_pushes = 0;
long g_pops = 0;
std::string g_last;

}

extern "C" {

__attribute__((visibility("default"))) int roctxRangePushA(const char* message) {
    g_last = message;
    return int(++g_pushes);
}

__attribute__((visibility("default"))) int roctxRangePop() {
    ++g_pops;
    return 0;
}

__attribute__((visibility("default"))) long FakeRoctxPushes() {
    return g_pushes;
}

__attribute__((visibility("default"))) long FakeRoctxPops() {
    return g_pops;
}

__attribute__((visibility("default"))) const char* FakeRoctxLastRange() {
    return g_last.c_str();
}
}

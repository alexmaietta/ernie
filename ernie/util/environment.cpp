// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/environment.h"

#include <cstdlib>

#include "ernie/util/error.h"

namespace ernie {

std::string Environment(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? value : "";
}

void ThrowUnknownChoice(const char* name, std::string_view value,
                        const std::vector<std::string>& names) {
    const std::string alternatives =
        names.size() < 2 ? Join(names)
                         : Join({names.begin(), names.end() - 1}) + " or " + names.back();
    throw Error("{} is \"{}\"; it may be {}", name, value, alternatives);
}

}

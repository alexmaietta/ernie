# SPDX-License-Identifier: BSD-3-Clause
# The demos, `<category>/demo_<category>_<name>.cpp`, each registering itself with
# ERNIE_REGISTER_DEMO; an object library, so the program and tests linking it keep every one.
add_library(ernie_demos OBJECT basics/demo_basics_bounce.cpp)
target_link_libraries(ernie_demos PUBLIC ernie_viewer PRIVATE ernie_build_options)
target_link_libraries(ernie_app PRIVATE ernie_demos)

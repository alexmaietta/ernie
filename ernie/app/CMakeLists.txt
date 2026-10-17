# SPDX-License-Identifier: BSD-3-Clause
add_executable(ernie_app main.cpp)
set_target_properties(ernie_app PROPERTIES OUTPUT_NAME ernie)
target_link_libraries(ernie_app PRIVATE ernie ernie_build_options)
install(TARGETS ernie_app RUNTIME DESTINATION bin COMPONENT app)

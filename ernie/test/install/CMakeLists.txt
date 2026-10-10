# SPDX-License-Identifier: BSD-3-Clause
# An application built against an installed Ernie, as users build theirs: find_package(Ernie) and
# link Ernie::ernie. Configured and run by run_install_test.cmake.
cmake_minimum_required(VERSION 3.22)
project(ErnieInstallConsumer LANGUAGES CXX)

find_package(Ernie 0.1 REQUIRED CONFIG)

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE Ernie::ernie)

# The README's C++ example, as written (extracted by run_install_test.cmake).
if(README_CPP)
    add_executable(readme_example "${README_CPP}")
    target_link_libraries(readme_example PRIVATE Ernie::ernie)
endif()

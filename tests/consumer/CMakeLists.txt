# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.22)
project(ErnieConsumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ernie 0.1 REQUIRED CONFIG OPTIONAL_COMPONENTS vulkan)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE Ernie::ernie)

enable_testing()
add_test(NAME adapters COMMAND consumer)
add_test(
    NAME round_trip
    COMMAND
        consumer round-trip
)

if(TARGET Ernie::vulkan)
    add_executable(consumer_vulkan vulkan.cpp)
    target_link_libraries(consumer_vulkan PRIVATE Ernie::vulkan)
    add_test(NAME vulkan COMMAND consumer_vulkan)
endif()

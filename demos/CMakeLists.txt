# SPDX-License-Identifier: BSD-3-Clause
# Ernie's demos, by category: demos/<category>/<name>.cpp beside <name>.py, the same demo in each
# language (interop/ is Python only). Each C++ demo registers itself (ERNIE_REGISTER_DEMO), and
# their objects go into the `ernie` program and the Python module, whose Demos panels list them.
# With ERNIE_BUILD_DEMOS each is also a program of its own, demo_<category>_<name>.

file(GLOB ernie_demo_cpp CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp")

# An object library: a static archive would drop the demos' objects, which nothing references
# but their registrations.
add_library(ernie_demos OBJECT util.cpp ${ernie_demo_cpp})
target_include_directories(ernie_demos
                           PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ernie_demos PRIVATE ernie_app
                                          ernie_build_options)
set_target_properties(ernie_demos PROPERTIES POSITION_INDEPENDENT_CODE
                                             ON)

target_sources(ernie_viewer PRIVATE $<TARGET_OBJECTS:ernie_demos>)
if(TARGET _ernie)
    target_sources(_ernie PRIVATE $<TARGET_OBJECTS:ernie_demos>)
endif()

if(ERNIE_BUILD_DEMOS)
    foreach(source ${ernie_demo_cpp})
        get_filename_component(name "${source}" NAME_WE)
        get_filename_component(directory "${source}" DIRECTORY)
        get_filename_component(category "${directory}" NAME)
        set(program "demo_${category}_${name}")
        add_executable(${program} main.cpp
                                  $<TARGET_OBJECTS:ernie_demos>)
        target_compile_definitions(
            ${program} PRIVATE ERNIE_DEMO_ID="${category}/${name}")
        target_link_libraries(${program} PRIVATE ernie_app
                                                 ernie_build_options)
        if(ERNIE_INSTALL)
            install(TARGETS ${program}
                    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                            COMPONENT demos)
        endif()
    endforeach()
endif()

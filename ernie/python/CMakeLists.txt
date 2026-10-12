# SPDX-License-Identifier: BSD-3-Clause
set(ernie_package_dir "${CMAKE_BINARY_DIR}/python/ernie")

nanobind_add_module(
    _ernie
    STABLE_ABI
    NB_STATIC
    module.cpp
    render.cpp
    scene.cpp
    scene_json.cpp
    schema.cpp
    tensor.cpp)
target_link_libraries(_ernie PRIVATE ernie ernie_build_options)
if(TARGET ernie_app)
    target_sources(_ernie PRIVATE app.cpp demo.cpp)
    target_link_libraries(_ernie PRIVATE ernie_app)
    target_compile_definitions(_ernie PRIVATE ERNIE_HAS_APP)
endif()
target_compile_features(_ernie PRIVATE cxx_std_20)
target_include_directories(
    _ernie SYSTEM PRIVATE "${NB_DIR}/include" ${Python_INCLUDE_DIRS})
foreach(nanobind_library nanobind-static nanobind-static-abi3)
    if(TARGET ${nanobind_library})
        set_target_properties(${nanobind_library}
                              PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    endif()
endforeach()
set_target_properties(_ernie PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                        "${ernie_package_dir}")
# Windows has no rpath: the module finds Slang's DLLs beside it, so the build's package imports
# (and its stubs generate).
if(WIN32)
    file(COPY ${ERNIE_SLANG_RUNTIME}
         DESTINATION "${ernie_package_dir}")
endif()
# The native `ernie` program beside the build's package too, as in the
# wheel, so the `ernie` command (ernie/__main__.py) runs from a build.
if(TARGET ernie_viewer)
    set(ernie_program
        "${ernie_package_dir}/ernie${CMAKE_EXECUTABLE_SUFFIX}")
    add_custom_command(
        OUTPUT "${ernie_program}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ernie_viewer> "${ernie_program}"
        DEPENDS ernie_viewer)
    add_custom_target(_ernie_program ALL DEPENDS "${ernie_program}")
endif()
# Type stubs beside the module, for IDEs, type checkers and the Python API docs: _ernie/ holds a
# .pyi per module (the extension still imports first), and py.typed marks the package as typed.
set(ernie_stubs "${ernie_package_dir}/_ernie/__init__.pyi"
                "${ernie_package_dir}/_ernie/tokens.pyi")
if(TARGET ernie_app)
    list(APPEND ernie_stubs "${ernie_package_dir}/_ernie/app.pyi")
endif()
nanobind_add_stub(
    _ernie_stub
    RECURSIVE
    MODULE
    ernie._ernie
    OUTPUT_PATH
    "${ernie_package_dir}"
    OUTPUT
    ${ernie_stubs}
    MARKER_FILE
    "${ernie_package_dir}/py.typed"
    PATTERN_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/stub_patterns.txt"
    PYTHON_PATH
    "${CMAKE_BINARY_DIR}/python"
    DEPENDS
    _ernie
    ernie/schema.py)
configure_file(ernie/__init__.py "${ernie_package_dir}/__init__.py"
               COPYONLY)
configure_file(ernie/app.py "${ernie_package_dir}/app.py" COPYONLY)
configure_file(ernie/batch.py "${ernie_package_dir}/batch.py"
               COPYONLY)
configure_file(ernie/demo.py "${ernie_package_dir}/demo.py" COPYONLY)
configure_file(ernie/schema.py "${ernie_package_dir}/schema.py"
               COPYONLY)
configure_file(ernie/__main__.py "${ernie_package_dir}/__main__.py"
               COPYONLY)
# ernie.integrations: Ernie as other simulators' backend, such as Newton's viewer.
file(
    GLOB_RECURSE ernie_integration_sources CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/ernie"
    "${CMAKE_CURRENT_SOURCE_DIR}/ernie/integrations/*.py")
foreach(source ${ernie_integration_sources})
    configure_file("ernie/${source}" "${ernie_package_dir}/${source}"
                   COPYONLY)
endforeach()
# ernie.demos: the Python demos (demos/<category>/<name>.py) and the ISDs they read, run by
# `python -m ernie.demos`.
file(
    GLOB ernie_demo_sources CONFIGURE_DEPENDS
    RELATIVE "${PROJECT_SOURCE_DIR}/demos"
    "${PROJECT_SOURCE_DIR}/demos/*/*.py")
set(ernie_demo_data
    "${PROJECT_SOURCE_DIR}/ernie/test/fixtures/isd/lroNacM1282557272RE.json"
    "${PROJECT_SOURCE_DIR}/ernie/test/fixtures/isd/lroNacRCalibration.csv"
    "${PROJECT_SOURCE_DIR}/ernie/test/fixtures/isd/lroWacPushFrame.json"
)
foreach(source ernie/demos/__init__.py ernie/demos/__main__.py)
    get_filename_component(name "${source}" NAME)
    configure_file("${source}" "${ernie_package_dir}/demos/${name}"
                   COPYONLY)
endforeach()
foreach(source ${ernie_demo_sources})
    configure_file("${PROJECT_SOURCE_DIR}/demos/${source}"
                   "${ernie_package_dir}/demos/${source}" COPYONLY)
endforeach()
foreach(source ${ernie_demo_data})
    get_filename_component(name "${source}" NAME)
    configure_file("${source}"
                   "${ernie_package_dir}/demos/data/${name}" COPYONLY)
endforeach()
# Stubs of the typed prim classes, which ernie.schema builds from the prim type registry at import
# where nanobind's stubgen cannot see them; -P keeps this directory's ernie/ off the path.
add_custom_command(
    OUTPUT "${ernie_package_dir}/schema.pyi"
    COMMAND
        ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${CMAKE_BINARY_DIR}/python" ${Python_EXECUTABLE}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/stub_schema.py"
        "${ernie_package_dir}/schema.pyi"
    DEPENDS _ernie stub_schema.py ernie/schema.py ernie/__init__.py
    COMMENT "Generating the typed prim classes' stubs"
    VERBATIM)
add_custom_target(_ernie_schema_stub ALL
                  DEPENDS "${ernie_package_dir}/schema.pyi")
# A wheel (pyproject.toml, scikit-build-core): the package, with Slang's compiler and the glslang
# library it loads at run time beside the module, found through $ORIGIN; the `python` component
# only, not dependencies' own install rules.
if(SKBUILD)
    install(TARGETS _ernie LIBRARY DESTINATION ernie COMPONENT python)
    install(
        FILES ernie/__init__.py
              ernie/app.py
              ernie/batch.py
              ernie/demo.py
              ernie/schema.py
              ernie/__main__.py
              "${ernie_package_dir}/schema.pyi"
              "${ernie_package_dir}/py.typed"
        DESTINATION ernie
        COMPONENT python)
    install(
        FILES ${ernie_stubs}
        DESTINATION ernie/_ernie
        COMPONENT python)
    install(
        FILES ernie/demos/__init__.py ernie/demos/__main__.py
        DESTINATION ernie/demos
        COMPONENT python)
    foreach(source ${ernie_integration_sources})
        get_filename_component(directory "${source}" DIRECTORY)
        install(
            FILES "ernie/${source}"
            DESTINATION ernie/${directory}
            COMPONENT python)
    endforeach()
    foreach(source ${ernie_demo_sources})
        get_filename_component(category "${source}" DIRECTORY)
        install(
            FILES "${PROJECT_SOURCE_DIR}/demos/${source}"
            DESTINATION ernie/demos/${category}
            COMPONENT python)
    endforeach()
    install(
        FILES ${ernie_demo_data}
        DESTINATION ernie/demos/data
        COMPONENT python)
    # Slang's runtime beside the module, found through the rpath (Windows looks beside the module
    # anyway); on macOS MoltenVK too, which Ernie loads from beside its module without a Vulkan
    # loader installed.
    if(APPLE)
        set_target_properties(_ernie PROPERTIES INSTALL_RPATH
                                                "@loader_path")
    else()
        set_target_properties(_ernie PROPERTIES INSTALL_RPATH
                                                "$ORIGIN")
    endif()
    include("${PROJECT_SOURCE_DIR}/cmake/ErnieLicenses.cmake")
    ernie_install_licenses(ernie/licenses python)
    install(
        FILES ${ERNIE_SLANG_RUNTIME}
        DESTINATION ernie
        COMPONENT python)
    if(APPLE)
        install(
            FILES "${ERNIE_MOLTENVK_LIBRARY}"
            DESTINATION ernie
            COMPONENT python)
    endif()
    # The native `ernie` program, beside the module and Slang's runtime (its rpath finds them
    # there, as Windows does beside it): the `ernie` command (ernie/__main__.py) runs it.
    if(TARGET ernie_viewer)
        if(APPLE)
            set_target_properties(
                ernie_viewer PROPERTIES INSTALL_RPATH "@loader_path")
        else()
            set_target_properties(ernie_viewer
                                  PROPERTIES INSTALL_RPATH "$ORIGIN")
        endif()
        install(TARGETS ernie_viewer RUNTIME DESTINATION ernie
                                             COMPONENT python)
    endif()
endif()

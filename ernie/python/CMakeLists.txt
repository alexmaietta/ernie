# SPDX-License-Identifier: BSD-3-Clause
include("${PROJECT_SOURCE_DIR}/cmake/third_party/nanobind.cmake")

set(ernie_python_root "${PROJECT_BINARY_DIR}/python")
set(ernie_python_dir "${ernie_python_root}/ernie")
set(ernie_python_files
    __init__.py
    __main__.py
    _typed.py
    py.typed
)

nanobind_add_module(
    _ernie
    STABLE_ABI
    NB_STATIC
    src/device.cpp
    src/dlpack.cpp
    src/importer.cpp
    src/io.cpp
    src/module.cpp
    src/render.cpp
    src/rotation.cpp
    src/scene.cpp
    src/values.cpp
    "${ernie_python_defines}"
)
target_link_libraries(
    _ernie
    PRIVATE
        ernie
        ernie_app
)
ernie_warnings(_ernie)
if(MSVC)
    target_compile_options(_ernie PRIVATE /wd4714)
endif()
foreach(nanobind IN ITEMS nanobind-static nanobind-static-abi3)
    if(TARGET ${nanobind})
        set_target_properties(
            ${nanobind}
            PROPERTIES
                SYSTEM
                    ON
        )
    endif()
endforeach()
ernie_coverage(_ernie)
target_include_directories(_ernie PRIVATE src)
target_include_directories(_ernie SYSTEM PRIVATE ${Python_INCLUDE_DIRS})
set(ernie_python_version "${PROJECT_VERSION}")
if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
    set(ernie_python_version "${SKBUILD_PROJECT_VERSION_FULL}")
endif()
target_compile_definitions(_ernie PRIVATE ERNIE_VERSION="${ernie_python_version}")
set_target_properties(
    _ernie
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "$<1:${ernie_python_dir}>"
        DEBUG_POSTFIX
            ""
        INSTALL_RPATH
            "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,$ORIGIN>"
)
if(WIN32)
    add_custom_command(
        TARGET _ernie
        POST_BUILD
        COMMAND
            "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:ernie>" ${ERNIE_SLANG_RUNTIME}
            "${ernie_python_dir}"
    )
endif()

foreach(file IN LISTS ernie_python_files)
    configure_file("ernie/${file}" "${ernie_python_dir}/${file}" COPYONLY)
endforeach()

nanobind_add_stub(
    ernie_python_stubs_ernie
    MODULE ernie._ernie
    OUTPUT "${ernie_python_dir}/_ernie.pyi"
    PYTHON_PATH "${ernie_python_root}"
    DEPENDS _ernie
)

install(
    TARGETS
        _ernie
    LIBRARY
        DESTINATION ernie
        COMPONENT python
)
list(TRANSFORM ernie_python_files PREPEND "ernie/" OUTPUT_VARIABLE ernie_python_sources)
install(
    FILES
        ${ernie_python_sources}
        "${ernie_python_dir}/schemas.py"
        "${ernie_python_dir}/_ernie.pyi"
    DESTINATION ernie
    COMPONENT python
)
if(SKBUILD)
    set_property(TARGET ernie PROPERTY VERSION)
    set_property(TARGET ernie PROPERTY SOVERSION)
endif()
install(
    TARGETS
        ernie
    RUNTIME
        DESTINATION ernie
        COMPONENT python
    LIBRARY
        DESTINATION ernie
        COMPONENT python
        NAMELINK_SKIP
)
install(PROGRAMS ${ERNIE_SLANG_RUNTIME} DESTINATION ernie COMPONENT python)

if(ERNIE_BUILD_VIEWER)
    nanobind_add_module(
        _viewer
        STABLE_ABI
        NB_STATIC
        src/imgui.cpp
        src/viewer.cpp
    )
    target_link_libraries(
        _viewer
        PRIVATE
            ernie_viewer
            ernie
    )
    ernie_warnings(_viewer)
    if(MSVC)
        target_compile_options(_viewer PRIVATE /wd4714)
    endif()
    target_include_directories(_viewer PRIVATE src)
    target_include_directories(_viewer SYSTEM PRIVATE ${Python_INCLUDE_DIRS})
    set_target_properties(
        _viewer
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY
                "$<1:${ernie_python_dir}>"
            DEBUG_POSTFIX
                ""
            INSTALL_RPATH
                "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,$ORIGIN>"
    )
    if(WIN32)
        add_custom_command(
            TARGET _viewer
            POST_BUILD
            COMMAND
                "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:ernie_viewer>"
                "${ernie_python_dir}"
        )
    endif()
    configure_file(ernie/viewer.py "${ernie_python_dir}/viewer.py" COPYONLY)
    nanobind_add_stub(
        ernie_python_stubs_viewer
        MODULE ernie._viewer
        OUTPUT "${ernie_python_dir}/_viewer.pyi"
        PYTHON_PATH "${ernie_python_root}"
        DEPENDS
            _viewer
            _ernie
    )
    install(
        TARGETS
            _viewer
        LIBRARY
            DESTINATION ernie
            COMPONENT python
    )
    install(
        FILES
            ernie/viewer.py
            "${ernie_python_dir}/_viewer.pyi"
        DESTINATION ernie
        COMPONENT python
    )
endif()

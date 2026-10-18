# SPDX-License-Identifier: BSD-3-Clause
include("${PROJECT_SOURCE_DIR}/cmake/third_party/nanobind.cmake")

# The package is assembled in the build tree, so tests import it with PYTHONPATH=<build>/python.
set(ernie_package_dir "${PROJECT_BINARY_DIR}/python/ernie")

nanobind_add_module(_ernie STABLE_ABI NB_STATIC module.cpp)
target_link_libraries(_ernie PRIVATE ernie_static ernie_build_options)
# Third-party headers as system headers, so our warnings do not apply to them.
target_include_directories(_ernie SYSTEM PRIVATE "${NB_DIR}/include" ${Python_INCLUDE_DIRS})
# Its file name is its import name, whatever debug postfix a dependency sets (Imath sets "_d").
set_target_properties(
    _ernie
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${ernie_package_dir}" DEBUG_POSTFIX ""
)
foreach(nanobind_library nanobind-static nanobind-static-abi3)
    if(TARGET ${nanobind_library})
        set_target_properties(${nanobind_library} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    endif()
endforeach()

configure_file(ernie/__init__.py "${ernie_package_dir}/__init__.py" COPYONLY)

install(TARGETS _ernie LIBRARY DESTINATION ernie COMPONENT python)
install(FILES ernie/__init__.py DESTINATION ernie COMPONENT python)

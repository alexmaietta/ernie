set(ernie_package_dir "${CMAKE_BINARY_DIR}/python/ernie")

nanobind_add_module(
    _ernie
    STABLE_ABI
    NB_STATIC
    module.cpp
    render.cpp
    scene.cpp
    tensor.cpp)
target_link_libraries(_ernie PRIVATE ernie ernie_build_options)
if(TARGET ernie_app)
    target_sources(_ernie PRIVATE app.cpp)
    target_link_libraries(_ernie PRIVATE ernie_app)
    target_compile_definitions(_ernie PRIVATE ERNIE_HAS_APP)
endif()
target_compile_features(_ernie PRIVATE cxx_std_20)
target_include_directories(_ernie SYSTEM PRIVATE "${NB_DIR}/include" ${Python_INCLUDE_DIRS})
foreach(nanobind_library nanobind-static nanobind-static-abi3)
    if(TARGET ${nanobind_library})
        set_target_properties(${nanobind_library} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    endif()
endforeach()
set_target_properties(_ernie PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${ernie_package_dir}")
# Type stubs beside the module, for IDEs, type checkers and the Python API docs: _ernie/ holds a
# .pyi per module (the extension still imports first), and py.typed marks the package as typed.
set(ernie_stubs "${ernie_package_dir}/_ernie/__init__.pyi" "${ernie_package_dir}/_ernie/tokens.pyi")
if(TARGET ernie_app)
    list(APPEND ernie_stubs "${ernie_package_dir}/_ernie/app.pyi")
endif()
nanobind_add_stub(
    _ernie_stub
    RECURSIVE
    MODULE ernie._ernie
    OUTPUT_PATH "${ernie_package_dir}"
    OUTPUT ${ernie_stubs}
    MARKER_FILE "${ernie_package_dir}/py.typed"
    PYTHON_PATH "${CMAKE_BINARY_DIR}/python"
    DEPENDS _ernie)
configure_file(ernie/__init__.py "${ernie_package_dir}/__init__.py" COPYONLY)
configure_file(ernie/app.py "${ernie_package_dir}/app.py" COPYONLY)
configure_file(ernie/batch.py "${ernie_package_dir}/batch.py" COPYONLY)
# A wheel (pyproject.toml, scikit-build-core): the package, with Slang's compiler and the glslang
# library it loads at run time beside the module, found through $ORIGIN; the `python` component
# only, not dependencies' own install rules.
if(SKBUILD)
    install(TARGETS _ernie LIBRARY DESTINATION ernie COMPONENT python)
    install(
        FILES ernie/__init__.py ernie/app.py ernie/batch.py "${ernie_package_dir}/py.typed"
        DESTINATION ernie
        COMPONENT python)
    install(
        FILES ${ernie_stubs}
        DESTINATION ernie/_ernie
        COMPONENT python)
    set_target_properties(_ernie PROPERTIES INSTALL_RPATH "$ORIGIN")
    include("${PROJECT_SOURCE_DIR}/cmake/ErnieLicenses.cmake")
    ernie_install_licenses(ernie/licenses python)
    install(
        FILES ${ERNIE_SLANG_RUNTIME}
        DESTINATION ernie
        COMPONENT python)
endif()

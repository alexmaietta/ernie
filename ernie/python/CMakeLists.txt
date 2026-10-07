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
configure_file(ernie/__init__.py "${ernie_package_dir}/__init__.py" COPYONLY)
configure_file(ernie/app.py "${ernie_package_dir}/app.py" COPYONLY)
configure_file(ernie/batch.py "${ernie_package_dir}/batch.py" COPYONLY)
if(SKBUILD)
    install(TARGETS _ernie LIBRARY DESTINATION ernie)
endif()

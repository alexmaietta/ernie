# SPDX-License-Identifier: BSD-3-Clause
# ernie_install_licenses(<destination> <component>): Ernie's LICENSE, and the license
# of every third-party library built into Ernie's binaries, as <destination>/third_party/<name>/.
# For the packages (ErnieInstall.cmake) and the wheel (ernie/python).
include_guard(GLOBAL)

include(FetchContent)

function(ernie_install_licenses destination component)
    install(
        FILES "${PROJECT_SOURCE_DIR}/LICENSE"
        DESTINATION "${destination}"
        COMPONENT ${component})

    # license(<name> <file>...): the first of the files that exists.
    macro(license name)
        set(found FALSE)
        foreach(file ${ARGN})
            if(NOT found AND EXISTS "${file}")
                install(
                    FILES "${file}"
                    DESTINATION "${destination}/third_party/${name}"
                    COMPONENT ${component})
                set(found TRUE)
            endif()
        endforeach()
        if(NOT found)
            message(
                WARNING
                    "Ernie: no license file found for ${name} (${ARGN})"
            )
        endif()
    endmacro()

    foreach(
        dependency
        slang
        vulkan_headers
        volk
        vma
        glm
        openexr
        imath
        meshoptimizer
        nlohmann_json
        tracy
        nanobind
        dear_imgui
        implot
        imguizmo
        glfw
        ernie_font_inter_license
        ernie_font_lucide_license)
        FetchContent_GetProperties(${dependency})
    endforeach()

    license(slang "${slang_SOURCE_DIR}/LICENSE")
    license(vulkan-headers "${vulkan_headers_SOURCE_DIR}/LICENSE.md")
    license(volk "${volk_SOURCE_DIR}/LICENSE.md")
    license(vulkan-memory-allocator "${vma_SOURCE_DIR}/LICENSE.txt")
    license(glm "${glm_SOURCE_DIR}/copying.txt")
    license(dlpack "${PROJECT_SOURCE_DIR}/third_party/dlpack/LICENSE")
    license(stb "${PROJECT_SOURCE_DIR}/third_party/stb/LICENSE")
    # Not a dependency: the source Ernie's CAHVORE rays follow (third_party/README.md, Derived).
    license(
        cameramodelutilsjs
        "${PROJECT_SOURCE_DIR}/third_party/cameramodelutilsjs/LICENSE"
    )
    license(openexr "${openexr_SOURCE_DIR}/LICENSE.md")
    license(imath "${imath_SOURCE_DIR}/LICENSE.md")
    # Vendored by OpenEXR and built into it (openexr.cmake).
    license(libdeflate
            "${openexr_SOURCE_DIR}/external/deflate/COPYING")
    license(zstd "${openexr_SOURCE_DIR}/external/zstd/LICENSE")
    license(openjph "${openexr_SOURCE_DIR}/external/OpenJPH/LICENSE")
    license(meshoptimizer "${meshoptimizer_SOURCE_DIR}/LICENSE.md")
    license(nlohmann-json "${nlohmann_json_SOURCE_DIR}/LICENSE.MIT")
    if(APPLE)
        FetchContent_GetProperties(moltenvk)
        license(moltenvk "${moltenvk_SOURCE_DIR}/LICENSE"
                "${moltenvk_SOURCE_DIR}/MoltenVK/LICENSE")
    endif()
    if(ERNIE_TRACY)
        license(tracy "${tracy_SOURCE_DIR}/LICENSE")
    endif()
    # The Python module links nanobind statically (NB_STATIC): fetched, or pip's (NB_DIR is the
    # package, its license in the .dist-info beside it).
    if(component STREQUAL "python")
        set(nanobind_licenses "${nanobind_SOURCE_DIR}/LICENSE")
        if(NB_DIR)
            file(GLOB nanobind_licenses_pip "${NB_DIR}/LICENSE"
                 "${NB_DIR}/../nanobind-*.dist-info/licenses/LICENSE"
                 "${NB_DIR}/../nanobind-*.dist-info/LICENSE")
            list(APPEND nanobind_licenses ${nanobind_licenses_pip})
        endif()
        license(nanobind ${nanobind_licenses})
    endif()
    if(TARGET ernie_app)
        license(dear-imgui "${dear_imgui_SOURCE_DIR}/LICENSE.txt")
        license(implot "${implot_SOURCE_DIR}/LICENSE")
        license(imguizmo "${imguizmo_SOURCE_DIR}/LICENSE")
        license(inter-font
                "${ernie_font_inter_license_SOURCE_DIR}/LICENSE.txt")
        license(lucide-font
                "${ernie_font_lucide_license_SOURCE_DIR}/LICENSE")
        # A system GLFW is a dependency rather than built in.
        if(glfw_SOURCE_DIR)
            license(glfw "${glfw_SOURCE_DIR}/LICENSE.md")
        endif()
    endif()
endfunction()

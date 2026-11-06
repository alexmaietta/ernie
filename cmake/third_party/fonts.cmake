# SPDX-License-Identifier: BSD-3-Clause
include_guard(GLOBAL)
FetchContent_Declare(
    inter_font
    URL
        https://raw.githubusercontent.com/rsms/inter/v4.1/docs/font-files/InterVariable.ttf
    URL_HASH SHA256=4989b125924991b90d05b2d16e0e388c48f7d5bb8b30539bbf9c755278d0ccaf
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_Declare(
    lucide_font
    URL
        https://cdn.jsdelivr.net/npm/lucide-static@1.52.0/font/lucide.ttf
    URL_HASH SHA256=124ecc64b91a158fc519eefdaaead89a2d9e2c39b6f9c8429c5584c93a0be451
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_Declare(
    inter_license
    URL
        https://raw.githubusercontent.com/rsms/inter/v4.1/LICENSE.txt
    URL_HASH SHA256=262481e844521b326f5ecd053e59b98c8b2da78c8ee1bdbb6e8174305e54935a
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_Declare(
    lucide_license
    URL
        https://cdn.jsdelivr.net/npm/lucide-static@1.52.0/LICENSE
    URL_HASH SHA256=b495047bd93a9b06913511076f504daba17d5bbeb3e0650f3bb53a4220329c57
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(
    inter_font
    lucide_font
    inter_license
    lucide_license
)

set(font_sources)
foreach(
    font
    "inter|${inter_font_SOURCE_DIR}/InterVariable.ttf"
    "lucide|${lucide_font_SOURCE_DIR}/lucide.ttf"
)
    string(
        REPLACE "|"
        ";"
        parts
        "${font}"
    )
    list(GET parts 0 name)
    list(GET parts 1 file)
    set(output "${CMAKE_CURRENT_BINARY_DIR}/fonts/${name}_font.cpp")
    add_custom_command(
        OUTPUT
            "${output}"
        COMMAND
            "${CMAKE_COMMAND}" "-DINPUT=${file}" "-DOUTPUT=${output}" "-DSYMBOL=kErnieFont_${name}"
            -P "${PROJECT_SOURCE_DIR}/cmake/ErnieEmbed.cmake"
        DEPENDS
            "${file}"
            "${PROJECT_SOURCE_DIR}/cmake/ErnieEmbed.cmake"
        COMMENT "Embedding the ${name} font"
        VERBATIM
    )
    list(APPEND font_sources "${output}")
endforeach()
add_library(ernie_fonts STATIC ${font_sources})
set_target_properties(
    ernie_fonts
    PROPERTIES
        POSITION_INDEPENDENT_CODE
            ON
)

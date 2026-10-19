# SPDX-License-Identifier: BSD-3-Clause
# The viewer's typefaces, as Ernie 1's: Inter (SIL OFL 1.1) for text and Lucide (ISC) for icons,
# each a single TTF compiled in as a byte array (`kErnieFont_inter`, `kErnieFont_lucide`).
set(ERNIE_DEP_VERSION_inter v4.1 CACHE STRING "Tag of the Inter typeface to fetch")
set(ERNIE_DEP_VERSION_lucide 1.52.0 CACHE STRING "Version of lucide-static's icon font to fetch")
mark_as_advanced(ERNIE_DEP_VERSION_inter ERNIE_DEP_VERSION_lucide)

set(ernie_fonts_dir "${ERNIE_DEPS_DIR}/fonts")
set(ernie_fonts_generated "${CMAKE_BINARY_DIR}/generated/fonts")

# Downloads `url` to `file` in the fonts directory, checked against `hash`, unless it is there.
function(ernie_fetch_font file url hash)
    set(path "${ernie_fonts_dir}/${file}")
    if(NOT EXISTS "${path}")
        file(DOWNLOAD "${url}" "${path}" EXPECTED_HASH SHA256=${hash} STATUS status)
        list(GET status 0 code)
        if(NOT code EQUAL 0)
            file(REMOVE "${path}")
            message(FATAL_ERROR "Fetching ${url} failed: ${status}")
        endif()
    endif()
endfunction()

ernie_fetch_font(
    "InterVariable-${ERNIE_DEP_VERSION_inter}.ttf"
    "https://raw.githubusercontent.com/rsms/inter/${ERNIE_DEP_VERSION_inter}/docs/font-files/InterVariable.ttf"
    4989b125924991b90d05b2d16e0e388c48f7d5bb8b30539bbf9c755278d0ccaf
)
ernie_fetch_font(
    "inter-LICENSE-${ERNIE_DEP_VERSION_inter}.txt"
    "https://raw.githubusercontent.com/rsms/inter/${ERNIE_DEP_VERSION_inter}/LICENSE.txt"
    262481e844521b326f5ecd053e59b98c8b2da78c8ee1bdbb6e8174305e54935a
)
ernie_fetch_font(
    "lucide-${ERNIE_DEP_VERSION_lucide}.ttf"
    "https://cdn.jsdelivr.net/npm/lucide-static@${ERNIE_DEP_VERSION_lucide}/font/lucide.ttf"
    124ecc64b91a158fc519eefdaaead89a2d9e2c39b6f9c8429c5584c93a0be451
)
ernie_fetch_font(
    "lucide-LICENSE-${ERNIE_DEP_VERSION_lucide}.txt"
    "https://cdn.jsdelivr.net/npm/lucide-static@${ERNIE_DEP_VERSION_lucide}/LICENSE"
    b495047bd93a9b06913511076f504daba17d5bbeb3e0650f3bb53a4220329c57
)

set(ernie_font_sources)
foreach(
    font
    "inter|InterVariable-${ERNIE_DEP_VERSION_inter}.ttf"
    "lucide|lucide-${ERNIE_DEP_VERSION_lucide}.ttf"
)
    string(REPLACE "|" ";" parts "${font}")
    list(GET parts 0 name)
    list(GET parts 1 file)
    set(output "${ernie_fonts_generated}/${name}_font.cpp")
    add_custom_command(
        OUTPUT "${output}"
        COMMAND
            "${CMAKE_COMMAND}" "-DINPUT=${ernie_fonts_dir}/${file}" "-DOUTPUT=${output}"
            "-DSYMBOL=kErnieFont_${name}" -P "${PROJECT_SOURCE_DIR}/cmake/EmbedFile.cmake"
        DEPENDS "${ernie_fonts_dir}/${file}" "${PROJECT_SOURCE_DIR}/cmake/EmbedFile.cmake"
        COMMENT "Embedding the ${name} font"
        VERBATIM
    )
    list(APPEND ernie_font_sources "${output}")
endforeach()
add_library(ernie_fonts STATIC ${ernie_font_sources})
set_target_properties(ernie_fonts PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(ernie_font_notices
    "inter|${ernie_fonts_dir}/inter-LICENSE-${ERNIE_DEP_VERSION_inter}.txt"
    "lucide|${ernie_fonts_dir}/lucide-LICENSE-${ERNIE_DEP_VERSION_lucide}.txt"
)

include("${CMAKE_CURRENT_LIST_DIR}/inter_font.cmake")

set(ERNIE_DEP_VERSION_lucide
    1.52.0
    CACHE STRING
          "Version of the Lucide icon font (lucide-static) to fetch.")
set(ERNIE_DEP_HASH_lucide
    124ecc64b91a158fc519eefdaaead89a2d9e2c39b6f9c8429c5584c93a0be451
    CACHE STRING "SHA-256 of lucide.ttf at ERNIE_DEP_VERSION_lucide.")
mark_as_advanced(ERNIE_DEP_VERSION_lucide ERNIE_DEP_HASH_lucide)

include(FetchContent)

# Lucide (ISC) for icons, a single TTF; the text is Inter (inter_font.cmake).
FetchContent_Declare(
    ernie_font_lucide
    URL "https://cdn.jsdelivr.net/npm/lucide-static@${ERNIE_DEP_VERSION_lucide}/font/lucide.ttf"
    URL_HASH SHA256=${ERNIE_DEP_HASH_lucide}
    DOWNLOAD_NO_EXTRACT TRUE)
# Its license, which goes with the font wherever it is redistributed (ErnieInstall.cmake).
FetchContent_Declare(
    ernie_font_lucide_license
    URL "https://cdn.jsdelivr.net/npm/lucide-static@${ERNIE_DEP_VERSION_lucide}/LICENSE"
    URL_HASH
        SHA256=b495047bd93a9b06913511076f504daba17d5bbeb3e0650f3bb53a4220329c57
    DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(ernie_font_lucide
                           ernie_font_lucide_license)

# The fonts are compiled into the GUI as byte arrays, so it needs no data files at run time.
set(ernie_gui_font_dir "${CMAKE_BINARY_DIR}/_deps/ernie_gui_fonts")
file(GLOB font_file "${ernie_font_lucide_SOURCE_DIR}/*.ttf")
set(font_source "${ernie_gui_font_dir}/lucide_font.cpp")
add_custom_command(
    OUTPUT "${font_source}"
    COMMAND
        "${CMAKE_COMMAND}" -DINPUT=${font_file}
        -DOUTPUT=${font_source} -DSYMBOL=kErnieFont_lucide -P
        "${CMAKE_CURRENT_LIST_DIR}/embed_file.cmake"
    DEPENDS "${font_file}"
            "${CMAKE_CURRENT_LIST_DIR}/embed_file.cmake"
    COMMENT "Embedding the lucide font"
    VERBATIM)
add_library(ernie_gui_fonts STATIC "${font_source}")
target_link_libraries(ernie_gui_fonts PUBLIC ernie_font_inter)

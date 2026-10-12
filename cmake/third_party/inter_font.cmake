include_guard(GLOBAL)

set(ERNIE_DEP_VERSION_inter
    v4.1
    CACHE STRING "Tag of the Inter typeface to fetch.")
set(ERNIE_DEP_HASH_inter
    4989b125924991b90d05b2d16e0e388c48f7d5bb8b30539bbf9c755278d0ccaf
    CACHE STRING
          "SHA-256 of InterVariable.ttf at ERNIE_DEP_VERSION_inter.")
mark_as_advanced(ERNIE_DEP_VERSION_inter ERNIE_DEP_HASH_inter)

include(FetchContent)

# Inter (SIL OFL 1.1), a single TTF: overlay text, baked to distance fields, and the viewer's text.
FetchContent_Declare(
    ernie_font_inter
    URL "https://raw.githubusercontent.com/rsms/inter/${ERNIE_DEP_VERSION_inter}/docs/font-files/InterVariable.ttf"
    URL_HASH SHA256=${ERNIE_DEP_HASH_inter}
    DOWNLOAD_NO_EXTRACT TRUE)
# Its license, which goes with the font wherever it is redistributed (ErnieLicenses.cmake).
FetchContent_Declare(
    ernie_font_inter_license
    URL "https://raw.githubusercontent.com/rsms/inter/${ERNIE_DEP_VERSION_inter}/LICENSE.txt"
    URL_HASH
        SHA256=262481e844521b326f5ecd053e59b98c8b2da78c8ee1bdbb6e8174305e54935a
    DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(ernie_font_inter ernie_font_inter_license)

# Compiled in as a byte array, `kErnieFont_inter`, so nothing reads font files at run time.
file(GLOB ernie_font_inter_file
     "${ernie_font_inter_SOURCE_DIR}/*.ttf")
set(ernie_font_inter_source
    "${CMAKE_BINARY_DIR}/_deps/ernie_gui_fonts/inter_font.cpp")
add_custom_command(
    OUTPUT "${ernie_font_inter_source}"
    COMMAND
        "${CMAKE_COMMAND}" -DINPUT=${ernie_font_inter_file}
        -DOUTPUT=${ernie_font_inter_source} -DSYMBOL=kErnieFont_inter
        -P "${CMAKE_CURRENT_LIST_DIR}/embed_file.cmake"
    DEPENDS "${ernie_font_inter_file}"
            "${CMAKE_CURRENT_LIST_DIR}/embed_file.cmake"
    COMMENT "Embedding the inter font"
    VERBATIM)
add_library(ernie_font_inter STATIC "${ernie_font_inter_source}")
set_target_properties(ernie_font_inter
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

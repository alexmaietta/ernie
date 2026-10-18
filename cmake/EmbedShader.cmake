# SPDX-License-Identifier: BSD-3-Clause
# Embeds one Slang file in a header as an ernie::ShaderModule
file(READ "${INPUT}" source)
string(FIND "${source}" ")slang\"" delimiter)
if(NOT delimiter EQUAL -1)
    message(FATAL_ERROR "${INPUT} holds )slang\", which ends the raw string it is embedded in")
endif()

string(REPLACE "_" ";" words "${NAME}")
set(identifier "k")
foreach(word IN LISTS words)
    string(SUBSTRING "${word}" 0 1 first)
    string(SUBSTRING "${word}" 1 -1 rest)
    string(TOUPPER "${first}" first)
    string(APPEND identifier "${first}${rest}")
endforeach()

set(chunks "")
string(LENGTH "${source}" length)
# MSVC limits a string literal to 16 KB; concatenated pieces may total any length.
set(chunk 8192)
set(at 0)
while(at LESS length)
    string(SUBSTRING "${source}" ${at} ${chunk} piece)
    string(APPEND chunks "    R\"slang(${piece})slang\"\n")
    math(EXPR at "${at} + ${chunk}")
endwhile()
if(chunks STREQUAL "")
    set(chunks "    \"\"\n")
endif()

file(RELATIVE_PATH from "${ROOT}" "${INPUT}")
# A file with a part C++ reads shares its types: the header brings them too.
set(shared "")
string(FIND "${source}" "#ifdef __cplusplus" cplusplus)
if(NOT cplusplus EQUAL -1)
    set(shared "#include \"${from}\"\n")
endif()
set(header
    "// Generated from ${from} by cmake/EmbedShader.cmake: edit that file.\n#pragma once\n\n#include \"ernie/device/shader_compiler.h\"\n${shared}\nnamespace ernie::shaders {\n\ninline const ShaderModule ${identifier}{\n    \"${NAME}\",\n${chunks}};\n\n}  // namespace ernie::shaders\n"
)
# Unchanged headers keep their times, so nothing that includes them rebuilds.
if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" previous)
    if(previous STREQUAL header)
        return()
    endif()
endif()
file(WRITE "${OUTPUT}" "${header}")

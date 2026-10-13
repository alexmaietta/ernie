# Runs OpenUSD's build_usd.py into USD_INSTALL once (cmake -P, from openusd.cmake's
# ExternalProject), under a lock on the install directory: builds of the same tree started at once
# (two targets, two sessions on one build directory) wait for each other, and the later ones find
# it built.
#
# Takes PYTHON, USD_SOURCE, USD_WORK, USD_INSTALL, USD_JOBS and USD_ARGS (a list, its separator
# "|").

file(MAKE_DIRECTORY "${USD_INSTALL}")
file(LOCK "${USD_INSTALL}.lock" GUARD PROCESS TIMEOUT 14400 RESULT_VARIABLE lock_result)
if(NOT lock_result EQUAL 0)
    message(FATAL_ERROR "Cannot lock ${USD_INSTALL}.lock: ${lock_result}")
endif()
if(EXISTS "${USD_INSTALL}/.built")
    message(STATUS "OpenUSD is built in ${USD_INSTALL}")
    return()
endif()
string(REPLACE "|" ";" args "${USD_ARGS}")
execute_process(
    COMMAND "${PYTHON}" "${USD_SOURCE}/build_scripts/build_usd.py" ${args} --src "${USD_WORK}/src"
            --build "${USD_WORK}/build" --jobs ${USD_JOBS} "${USD_INSTALL}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Building OpenUSD failed; see ${USD_WORK}/build/*/log.txt")
endif()
file(TOUCH "${USD_INSTALL}/.built")

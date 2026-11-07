# SPDX-License-Identifier: BSD-3-Clause
file(REMOVE_RECURSE "${BINARY}")
execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -S "${SOURCE}" -B "${BINARY}" "-DERNIE_ROOT=${ROOT}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)
if(result EQUAL 0)
    message(FATAL_ERROR "${SOURCE} configured, but it should have failed")
endif()
if(NOT "${output}${error}" MATCHES "${EXPECTED}")
    message(FATAL_ERROR "${SOURCE} failed without \"${EXPECTED}\":\n${output}${error}")
endif()

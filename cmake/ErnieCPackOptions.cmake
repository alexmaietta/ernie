# SPDX-License-Identifier: BSD-3-Clause
# Per-generator CPack settings, read by cpack for each generator (CPACK_PROJECT_CONFIG_FILE in
# ErniePackaging.cmake): archives unpack into one top-level directory.
if(CPACK_GENERATOR MATCHES "TGZ|TXZ|ZIP")
    set(CPACK_COMPONENT_INCLUDE_TOPLEVEL_DIRECTORY ON)
endif()

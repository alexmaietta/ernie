# SPDX-License-Identifier: BSD-3-Clause
add_library(
    ernie_physics
    STATIC
    astronomy.cpp
    atmosphere.cpp
    cloud.cpp
    reflectance_library.cpp
    regolith.cpp
    scattering.cpp
    solar.cpp
    spectrum.cpp
)
target_link_libraries(ernie_physics PUBLIC ernie_util PRIVATE ernie_build_options)

# The shaders, embedded as strings into ernie/kernel/embedded_shaders.h.
set(ernie_shaders
    closure/brdf.slang
    closure/hapke.slang
    closure/lambert.slang
    closure/lommel_seeliger.slang
    closure/lunar_lambert.slang
    closure/oren_nayar.slang
    film/sensor.slang
    raster/cull.slang
    raster/hzb.slang
    raster/frame.slang
    raster/ground.slang
    raster/mesh.slang
    raster/present.slang
    raster/shading.slang
    raster/surface.slang
    raster/terrain.slang
    raster/terrain_color.slang
    raster/terrain_shadow.slang
    raster/views.slang)
foreach(shader ${ernie_shaders})
    string(MAKE_C_IDENTIFIER "${shader}" shader_id)
    string(TOUPPER "${shader_id}" shader_upper)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${shader}" ERNIE_${shader_upper})
    set_property(
        DIRECTORY
        APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${shader}")
endforeach()
configure_file(embedded_shaders.h.in
               "${PROJECT_BINARY_DIR}/generated/ernie/kernel/embedded_shaders.h" @ONLY)

# C++ demos, each an executable named after its file: demo_<category>_<description>.

add_executable(demo_worlds_batched_clones demo_worlds_batched_clones.cpp)
target_link_libraries(demo_worlds_batched_clones PRIVATE ernie ernie_stb)

if(TARGET ernie_app)
    add_executable(demo_viewer_sun_panel demo_viewer_sun_panel.cpp)
    target_link_libraries(demo_viewer_sun_panel PRIVATE ernie_app)
endif()

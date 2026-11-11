// SPDX-License-Identifier: BSD-3-Clause
#include "mgr.hpp"

#include <madrona/py/bindings.hpp>

namespace nb = nanobind;

namespace ernie_bench {

using Floats = nb::ndarray<const float, nb::device::cpu, nb::c_contig>;
using Indices = nb::ndarray<const uint32_t, nb::device::cpu, nb::c_contig>;
using Ids = nb::ndarray<const int32_t, nb::device::cpu, nb::c_contig>;

static void require(bool condition, const char* message) {
    if (!condition) {
        throw nb::value_error(message);
    }
}

NB_MODULE(_madrona_bench, m) {
    nb::class_<Manager>(m, "Renderer")
        .def("__init__",
             [](Manager* self, int gpu_id, uint32_t world_count, uint32_t width, uint32_t height,
                bool raytrace, float fov_y, float near_plane, float far_plane,
                Floats light_direction, Floats vertices, Floats normals, Indices triangles,
                Indices vertex_offsets, Indices triangle_offsets, Ids object_meshes,
                Floats object_colors, Ids instance_objects, Indices world_offsets,
                uint32_t camera_count) {
                 require(light_direction.size() == 3, "light_direction needs 3 values");
                 require(vertices.size() == normals.size(), "vertices and normals differ in size");
                 require(object_colors.size() == 3 * object_meshes.size(),
                         "object_colors needs 3 values per object");
                 require(world_offsets.size() == (size_t)world_count + 1,
                         "world_offsets needs one entry per world plus the end");
                 require(world_offsets.data()[world_count] == instance_objects.size(),
                         "world_offsets must end at the instance count");
                 require(!raytrace || width == height,
                         "the madrona ray tracer only renders square images");
                 const float* light = light_direction.data();
                 new (self) Manager(
                     Manager::Config{
                         .gpuID = gpu_id,
                         .worldCount = world_count,
                         .width = width,
                         .height = height,
                         .useRT = raytrace,
                         .fovY = fov_y,
                         .nearPlane = near_plane,
                         .farPlane = far_plane,
                         .lightDirection = {light[0], light[1], light[2]},
                     },
                     Geometry{
                         .vertices = vertices.data(),
                         .normals = normals.data(),
                         .indices = triangles.data(),
                         .vertexOffsets = vertex_offsets.data(),
                         .triangleOffsets = triangle_offsets.data(),
                         .vertexCount = (uint32_t)(vertices.size() / 3),
                         .triangleCount = (uint32_t)(triangles.size() / 3),
                         .meshCount = (uint32_t)vertex_offsets.size(),
                         .objectMeshes = object_meshes.data(),
                         .objectColors = object_colors.data(),
                         .objectCount = (uint32_t)object_meshes.size(),
                     },
                     instance_objects.data(), world_offsets.data(), camera_count);
             })
        .def("set_instances",
             [](Manager& mgr, Floats positions, Floats rotations, Floats scales) {
                 size_t count = mgr.instanceCount();
                 require(positions.size() == 3 * count && rotations.size() == 4 * count &&
                             scales.size() == 3 * count,
                         "instance arrays must hold every instance of every world");
                 mgr.setInstances(positions.data(), rotations.data(), scales.data());
             })
        .def("set_cameras",
             [](Manager& mgr, Floats positions, Floats rotations) {
                 size_t count = (size_t)mgr.worldCount() * mgr.cameraCount();
                 require(positions.size() == 3 * count && rotations.size() == 4 * count,
                         "camera arrays must hold every camera of every world");
                 mgr.setCameras(positions.data(), rotations.data());
             })
        .def("render", &Manager::render, nb::call_guard<nb::gil_scoped_release>())
        .def("synchronize", &Manager::synchronize, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("stream", &Manager::stream)
        .def("read_color",
             [](const Manager& mgr, nb::ndarray<uint8_t, nb::device::cpu, nb::c_contig> out) {
                 require(out.size() == 4 * mgr.pixelCount(),
                         "color output needs 4 bytes per pixel of every view");
                 mgr.readColor(out.data());
             })
        .def("read_depth",
             [](const Manager& mgr, nb::ndarray<float, nb::device::cpu, nb::c_contig> out) {
                 require(out.size() == mgr.pixelCount(),
                         "depth output needs one float per pixel of every view");
                 mgr.readDepth(out.data());
             });
}

}

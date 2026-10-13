// SPDX-License-Identifier: BSD-3-Clause
// Writes the OpenVDB test fixture
// ernie/test/fixtures/volumes/cube.vdb with the OpenVDB that Ernie
// builds (cmake/third_party/openvdb.cmake), Blosc-compressed as
// OpenVDB writes by default:
//
// - "density": 16^3 ones in 0.1 m voxels, index (0, 0, 0) at
//   (-0.75, -0.75, -0.75) m, as cube.nvdb
//   (tools/volume_fixtures.py), so the two render alike;
// - a second "density": twos, for fieldIndex 1;
// - "velocity": a vec3f grid of (0, 0, 2) over the same voxels.
//
// Build and run from the repository, with V the build's OpenVDB
// install (build/_deps/openvdb-<version>-<key>):
//
//   c++ -std=c++20 -O2 -DOPENVDB_STATICLIB -I$V/include \
//     tools/vdb_fixtures.cpp $V/lib/libopenvdb.a $V/lib/libtbb.a \
//     $V/lib/libblosc.a $V/lib/libz.a -lpthread -ldl \
//     -o /tmp/vdb_fixtures && /tmp/vdb_fixtures

#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>

template <typename GridT>
typename GridT::Ptr Cube(const std::string& name,
                         const typename GridT::ValueType& value) {
    typename GridT::Ptr grid = GridT::create();
    grid->setName(name);
    grid->setTransform(
        openvdb::math::Transform::createLinearTransform(0.1));
    grid->transform().postTranslate(openvdb::Vec3d(-0.75));
    auto accessor = grid->getAccessor();
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            for (int k = 0; k < 16; ++k) {
                accessor.setValue(openvdb::Coord(i, j, k), value);
            }
        }
    }
    return grid;
}

int main() {
    openvdb::initialize();
    openvdb::GridPtrVec grids;
    grids.push_back(Cube<openvdb::FloatGrid>("density", 1.0f));
    openvdb::FloatGrid::Ptr density =
        openvdb::gridPtrCast<openvdb::FloatGrid>(grids.back());
    density->setGridClass(openvdb::GRID_FOG_VOLUME);
    grids.push_back(Cube<openvdb::FloatGrid>("density", 2.0f));
    grids.push_back(Cube<openvdb::Vec3fGrid>(
        "velocity", openvdb::Vec3f(0.0f, 0.0f, 2.0f)));
    openvdb::io::File("ernie/test/fixtures/volumes/cube.vdb")
        .write(grids);
    return 0;
}

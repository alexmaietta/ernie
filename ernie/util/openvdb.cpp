// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/nanovdb.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ERNIE_HAS_OPENVDB)
// NanoVDB's conversion runs its loops on oneTBB, which OpenVDB links.
#define NANOVDB_USE_TBB
#include <nanovdb/tools/CreateNanoGrid.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>

#include <mutex>
#endif

namespace ernie {

std::vector<NanoVdbGrid>
ReadOpenVdb(const std::filesystem::path& path) {
#if defined(ERNIE_HAS_OPENVDB)
    static std::once_flag initialized;
    std::call_once(initialized, [] { openvdb::initialize(); });
    openvdb::GridPtrVecPtr grids;
    try {
        openvdb::io::File file(path.string());
        file.open();
        grids = file.getGrids();
        file.close();
    } catch (const std::exception& e) {
        throw std::runtime_error("cannot read OpenVDB file " +
                                 path.string() + ": " + e.what());
    }
    std::vector<NanoVdbGrid> out;
    out.reserve(grids->size());
    for (const openvdb::GridBase::Ptr& grid : *grids) {
        try {
            const nanovdb::GridHandle<nanovdb::HostBuffer> handle =
                nanovdb::tools::openToNanoVDB(grid);
            std::vector<NanoVdbGrid> converted =
                NanoVdbGridsOf(std::span<const std::uint8_t>(
                    static_cast<const std::uint8_t*>(handle.data()),
                    static_cast<std::size_t>(handle.size())));
            out.push_back(std::move(converted.front()));
        } catch (const std::exception&) {
            // A type NanoVDB lacks: in place, so indices still count.
            NanoVdbGrid unknown;
            unknown.name = grid->getName();
            out.push_back(std::move(unknown));
        }
    }
    return out;
#else
    throw std::runtime_error(
        path.string() +
        " is an OpenVDB file, which this build of Ernie (without "
        "ERNIE_OPENVDB) cannot read; convert it to NanoVDB (.nvdb)");
#endif
}

bool OpenVdbAvailable() {
#if defined(ERNIE_HAS_OPENVDB)
    return true;
#else
    return false;
#endif
}

std::vector<NanoVdbGrid>
ReadGrids(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(), [](char c) {
        return static_cast<char>(
            std::tolower(static_cast<unsigned char>(c)));
    });
    return extension == ".vdb" ? ReadOpenVdb(path)
                               : ReadNanoVdb(path);
}

}  // namespace ernie

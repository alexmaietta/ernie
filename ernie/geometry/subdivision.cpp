// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/subdivision.h"

#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/surface.h>
#include <opensubdiv/bfr/tessellation.h>
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/topologyRefinerFactory.h>
#include <opensubdiv/sdc/crease.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ernie {
namespace {

namespace osd = OpenSubdiv;
using osd::Far::ConstIndexArray;
using osd::Far::Index;

osd::Sdc::SchemeType SchemeOf(SubdivisionScheme scheme) {
    switch (scheme) {
    case SubdivisionScheme::Loop:
        return osd::Sdc::SCHEME_LOOP;
    case SubdivisionScheme::Bilinear:
        return osd::Sdc::SCHEME_BILINEAR;
    default:
        return osd::Sdc::SCHEME_CATMARK;
    }
}

osd::Sdc::Options OptionsOf(const SubdivisionSurface& surface) {
    osd::Sdc::Options options;
    switch (surface.boundary) {
    case BoundaryInterpolation::None:
        options.SetVtxBoundaryInterpolation(
            osd::Sdc::Options::VTX_BOUNDARY_NONE);
        break;
    case BoundaryInterpolation::EdgeOnly:
        options.SetVtxBoundaryInterpolation(
            osd::Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);
        break;
    case BoundaryInterpolation::EdgeAndCorner:
        options.SetVtxBoundaryInterpolation(
            osd::Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);
        break;
    }
    static constexpr std::array kFaceVarying{
        osd::Sdc::Options::FVAR_LINEAR_NONE,
        osd::Sdc::Options::FVAR_LINEAR_CORNERS_ONLY,
        osd::Sdc::Options::FVAR_LINEAR_CORNERS_PLUS1,
        osd::Sdc::Options::FVAR_LINEAR_CORNERS_PLUS2,
        osd::Sdc::Options::FVAR_LINEAR_BOUNDARIES,
        osd::Sdc::Options::FVAR_LINEAR_ALL};
    options.SetFVarLinearInterpolation(
        kFaceVarying[static_cast<std::size_t>(surface.faceVarying)]);
    options.SetTriangleSubdivision(
        surface.triangles == TriangleSubdivision::Smooth
            ? osd::Sdc::Options::TRI_SUB_SMOOTH
            : osd::Sdc::Options::TRI_SUB_CATMARK);
    return options;
}

int IndexOf(std::int64_t value, std::size_t count, const char* what) {
    if (value < 0 || static_cast<std::uint64_t>(value) >= count) {
        throw std::invalid_argument(std::string(what) +
                                    " has an index out of range");
    }
    return static_cast<int>(value);
}

/// The control mesh as OpenSubdiv's topology, its arrays kept alive
/// for `Create`.
struct Topology {
    std::vector<int> counts;
    std::vector<Index> indices;
    std::vector<Index> creasePairs;
    std::vector<float> creaseWeights;
    std::vector<Index> corners;
    std::vector<float> cornerWeights;
    std::vector<Index> holes;
    std::vector<Index> uvIndices;
    std::vector<float> uvValues;  ///< s, t per face-varying value.

    std::unique_ptr<osd::Far::TopologyRefiner>
    Refiner(const SubdivisionSurface& surface, int pointCount) const {
        osd::Far::TopologyDescriptor desc;
        desc.numVertices = pointCount;
        desc.numFaces = static_cast<int>(counts.size());
        desc.numVertsPerFace = counts.data();
        desc.vertIndicesPerFace = indices.data();
        desc.numCreases = static_cast<int>(creaseWeights.size());
        desc.creaseVertexIndexPairs = creasePairs.data();
        desc.creaseWeights = creaseWeights.data();
        desc.numCorners = static_cast<int>(corners.size());
        desc.cornerVertexIndices = corners.data();
        desc.cornerWeights = cornerWeights.data();
        desc.numHoles = static_cast<int>(holes.size());
        desc.holeIndices = holes.data();
        osd::Far::TopologyDescriptor::FVarChannel channel;
        if (!uvIndices.empty()) {
            channel.numValues = static_cast<int>(uvValues.size() / 2);
            channel.valueIndices = uvIndices.data();
            desc.numFVarChannels = 1;
            desc.fvarChannels = &channel;
        }
        using Factory = osd::Far::TopologyRefinerFactory<
            osd::Far::TopologyDescriptor>;
        std::unique_ptr<osd::Far::TopologyRefiner> refiner(
            Factory::Create(desc,
                            Factory::Options(SchemeOf(surface.scheme),
                                             OptionsOf(surface))));
        if (!refiner) {
            throw std::invalid_argument("OpenSubdiv cannot build the "
                                        "control mesh's topology");
        }
        return refiner;
    }
};

Topology TopologyOf(const SubdivisionSurface& surface) {
    const std::size_t pointCount = surface.points.size() / 3;
    Topology topology;
    std::size_t corners = 0;
    for (const std::int64_t count : surface.faceVertexCounts) {
        if (count < 3) {
            throw std::invalid_argument(
                "Every face needs at least 3 vertices");
        }
        if (surface.scheme == SubdivisionScheme::Loop && count != 3) {
            throw std::invalid_argument(
                "A Loop subdivision surface needs triangles only");
        }
        topology.counts.push_back(static_cast<int>(count));
        corners += static_cast<std::size_t>(count);
    }
    if (corners != surface.faceVertexIndices.size()) {
        throw std::invalid_argument(
            "faceVertexCounts do not add up to faceVertexIndices");
    }
    for (const std::int64_t index : surface.faceVertexIndices) {
        topology.indices.push_back(
            IndexOf(index, pointCount, "faceVertexIndices"));
    }
    // Crease chains become edges, with a sharpness per chain or per
    // edge.
    std::size_t chainEdges = 0;
    std::size_t first = 0;
    for (const std::int64_t length : surface.creaseLengths) {
        if (length < 2 || first + static_cast<std::size_t>(length) >
                              surface.creaseIndices.size()) {
            throw std::invalid_argument(
                "creaseLengths disagree with creaseIndices");
        }
        chainEdges += static_cast<std::size_t>(length - 1);
        first += static_cast<std::size_t>(length);
    }
    const bool perChain = surface.creaseSharpnesses.size() ==
                          surface.creaseLengths.size();
    if (!perChain && surface.creaseSharpnesses.size() != chainEdges) {
        throw std::invalid_argument("creaseSharpnesses needs one "
                                    "value per crease or per crease "
                                    "edge");
    }
    first = 0;
    std::size_t edge = 0;
    for (std::size_t chain = 0; chain < surface.creaseLengths.size();
         ++chain) {
        const auto length =
            static_cast<std::size_t>(surface.creaseLengths[chain]);
        for (std::size_t k = 0; k + 1 < length; ++k, ++edge) {
            topology.creasePairs.push_back(
                IndexOf(surface.creaseIndices[first + k], pointCount,
                        "creaseIndices"));
            topology.creasePairs.push_back(
                IndexOf(surface.creaseIndices[first + k + 1],
                        pointCount, "creaseIndices"));
            topology.creaseWeights.push_back(
                surface.creaseSharpnesses[perChain ? chain : edge]);
        }
        first += length;
    }
    if (surface.cornerIndices.size() !=
        surface.cornerSharpnesses.size()) {
        throw std::invalid_argument(
            "cornerSharpnesses needs one value per corner");
    }
    for (std::size_t i = 0; i < surface.cornerIndices.size(); ++i) {
        topology.corners.push_back(IndexOf(
            surface.cornerIndices[i], pointCount, "cornerIndices"));
        topology.cornerWeights.push_back(
            surface.cornerSharpnesses[i]);
    }
    for (const std::int64_t hole : surface.holeIndices) {
        topology.holes.push_back(IndexOf(
            hole, surface.faceVertexCounts.size(), "holeIndices"));
    }
    // Texture coordinates as face-varying values: equal ones at a
    // point are one value, so the surface stays smooth across faces
    // sharing them and splits only along seams.
    if (!surface.cornerUvs.empty()) {
        if (surface.cornerUvs.size() != 2 * corners) {
            throw std::invalid_argument(
                "Texture coordinates need s, t per face corner");
        }
        struct Key {
            Index point;
            std::uint32_t s, t;
            bool operator==(const Key&) const = default;
        };
        struct KeyHash {
            std::size_t operator()(const Key& key) const {
                return (static_cast<std::size_t>(key.point) *
                        73856093u) ^
                       (static_cast<std::size_t>(key.s) * 19349663u) ^
                       (static_cast<std::size_t>(key.t) * 83492791u);
            }
        };
        std::unordered_map<Key, Index, KeyHash> values;
        for (std::size_t c = 0; c < corners; ++c) {
            Key key{topology.indices[c], 0, 0};
            std::memcpy(&key.s, &surface.cornerUvs[2 * c],
                        sizeof(float));
            std::memcpy(&key.t, &surface.cornerUvs[2 * c + 1],
                        sizeof(float));
            const auto [it, added] = values.try_emplace(
                key,
                static_cast<Index>(topology.uvValues.size() / 2));
            if (added) {
                topology.uvValues.push_back(surface.cornerUvs[2 * c]);
                topology.uvValues.push_back(
                    surface.cornerUvs[2 * c + 1]);
            }
            topology.uvIndices.push_back(it->second);
        }
    }
    return topology;
}

/// Where each face's tessellation goes: its boundary points (shared
/// with its neighbours, or its own) and whether it evaluates them,
/// the first of its interior points, and the first of its triangles.
struct Layout {
    std::vector<int> faces;  ///< Faces with a limit surface.
    std::vector<std::size_t>
        boundaryStart;                    ///< Per entry of `faces`.
    std::vector<std::uint32_t> boundary;  ///< Output point indices.
    std::vector<std::uint8_t> owned;  ///< Per entry of `boundary`.
    std::vector<std::uint32_t> interiorStart;
    std::vector<std::size_t> triangleStart;
    std::size_t points = 0;
    std::size_t triangles = 0;
};

Layout LayoutOf(const osd::Far::TopologyRefiner& refiner,
                const osd::Bfr::RefinerSurfaceFactory<>& factory,
                osd::Sdc::SchemeType scheme, int rate) {
    const osd::Far::TopologyLevel& base = refiner.GetLevel(0);
    const float infinite = osd::Sdc::Crease::SHARPNESS_INFINITE;
    // Points on infinitely sharp edges and corners are not shared, so
    // each side keeps the normal of its own side.
    std::vector<bool> sharpEdge(
        static_cast<std::size_t>(base.GetNumEdges()));
    for (int e = 0; e < base.GetNumEdges(); ++e) {
        sharpEdge[static_cast<std::size_t>(e)] =
            base.GetEdgeSharpness(e) >= infinite ||
            base.IsEdgeNonManifold(e);
    }
    std::vector<bool> sharpVertex(
        static_cast<std::size_t>(base.GetNumVertices()));
    for (int v = 0; v < base.GetNumVertices(); ++v) {
        bool sharp = base.GetVertexSharpness(v) >= infinite ||
                     base.IsVertexNonManifold(v);
        for (const Index e : base.GetVertexEdges(v)) {
            sharp = sharp || sharpEdge[static_cast<std::size_t>(e)];
        }
        sharpVertex[static_cast<std::size_t>(v)] = sharp;
    }
    constexpr std::uint32_t kUnset =
        std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> vertexPoint(sharpVertex.size(),
                                           kUnset);
    std::vector<std::uint32_t> edgePoints(sharpEdge.size(), kUnset);
    std::vector<Index> edgeStart(sharpEdge.size(), -1);

    osd::Bfr::Tessellation::Options options;
    options.SetFacetSize(3);
    Layout layout;
    std::uint64_t next = 0;
    const auto take = [&](std::uint64_t count) {
        const std::uint64_t first = next;
        next += count;
        if (next >= kUnset) {
            throw std::invalid_argument(
                "The subdivided mesh has too many points");
        }
        return static_cast<std::uint32_t>(first);
    };
    for (int face = 0; face < base.GetNumFaces(); ++face) {
        if (!factory.FaceHasLimitSurface(face)) {
            continue;
        }
        const ConstIndexArray verts = base.GetFaceVertices(face);
        const ConstIndexArray edges = base.GetFaceEdges(face);
        const osd::Bfr::Tessellation tess(
            osd::Bfr::Parameterization(scheme, verts.size()), rate,
            options);
        layout.faces.push_back(face);
        layout.boundaryStart.push_back(layout.boundary.size());
        for (int i = 0; i < verts.size(); ++i) {
            const auto v = static_cast<std::size_t>(verts[i]);
            if (sharpVertex[v]) {
                layout.boundary.push_back(take(1));
                layout.owned.push_back(1);
            } else if (vertexPoint[v] == kUnset) {
                vertexPoint[v] = take(1);
                layout.boundary.push_back(vertexPoint[v]);
                layout.owned.push_back(1);
            } else {
                layout.boundary.push_back(vertexPoint[v]);
                layout.owned.push_back(0);
            }
            const int inner = rate - 1;
            if (inner <= 0) {
                continue;
            }
            const auto e = static_cast<std::size_t>(edges[i]);
            if (sharpEdge[e]) {
                const std::uint32_t own =
                    take(static_cast<std::uint64_t>(inner));
                for (int k = 0; k < inner; ++k) {
                    layout.boundary.push_back(
                        own + static_cast<std::uint32_t>(k));
                    layout.owned.push_back(1);
                }
            } else if (edgePoints[e] == kUnset) {
                edgePoints[e] =
                    take(static_cast<std::uint64_t>(inner));
                edgeStart[e] = verts[i];
                for (int k = 0; k < inner; ++k) {
                    layout.boundary.push_back(
                        edgePoints[e] +
                        static_cast<std::uint32_t>(k));
                    layout.owned.push_back(1);
                }
            } else {
                // The neighbour walked the edge from the same end, or
                // from the other.
                const bool same = edgeStart[e] == verts[i];
                for (int k = 0; k < inner; ++k) {
                    layout.boundary.push_back(
                        edgePoints[e] +
                        static_cast<std::uint32_t>(
                            same ? k : inner - 1 - k));
                    layout.owned.push_back(0);
                }
            }
        }
        layout.interiorStart.push_back(take(
            static_cast<std::uint64_t>(tess.GetNumInteriorCoords())));
        layout.triangleStart.push_back(layout.triangles);
        layout.triangles +=
            static_cast<std::size_t>(tess.GetNumFacets());
    }
    layout.points = static_cast<std::size_t>(next);
    return layout;
}

/// A face's coarser tessellations in terms of its finest one's
/// coordinates, alike for every face of a parameterization.
struct CoarseLevels {
    int numBoundary = 0;  ///< The finest tessellation's.
    int numCoords = 0;
    /// Per coarser level, finest first: its triangles' corners, as
    /// coordinates of the finest tessellation.
    std::vector<std::vector<int>> facets;
    /// Per coarser level and finest coordinate: the corners of the
    /// coarse triangle over it, or -1s for any.
    std::vector<std::vector<std::array<int, 3>>> over;
};

CoarseLevels CoarseLevelsOf(const osd::Bfr::Parameterization& param,
                            int rate) {
    osd::Bfr::Tessellation::Options options;
    options.SetFacetSize(3);
    const osd::Bfr::Tessellation fine(param, rate, options);
    CoarseLevels levels;
    levels.numBoundary = fine.GetNumBoundaryCoords();
    levels.numCoords = fine.GetNumCoords();
    std::vector<float> fineCoords(
        2 * static_cast<std::size_t>(levels.numCoords));
    fine.GetCoords(fineCoords.data());
    for (int coarseRate = rate / 2; coarseRate >= 1;
         coarseRate /= 2) {
        const osd::Bfr::Tessellation tess(param, coarseRate, options);
        std::vector<float> coords(
            2 * static_cast<std::size_t>(tess.GetNumCoords()));
        tess.GetCoords(coords.data());
        std::vector<int> local(
            3 * static_cast<std::size_t>(tess.GetNumFacets()));
        tess.GetFacets(local.data());
        // Each coarse coordinate is a finest one (the nearest, should
        // they round apart).
        std::vector<int> same(coords.size() / 2);
        for (std::size_t c = 0; c < same.size(); ++c) {
            float nearest = std::numeric_limits<float>::infinity();
            for (int f = 0; f < levels.numCoords; ++f) {
                const float du =
                    fineCoords[2 * static_cast<std::size_t>(f)] -
                    coords[2 * c];
                const float dv =
                    fineCoords[2 * static_cast<std::size_t>(f) + 1] -
                    coords[2 * c + 1];
                if (du * du + dv * dv < nearest) {
                    nearest = du * du + dv * dv;
                    same[c] = f;
                }
            }
        }
        std::vector<int>& facets = levels.facets.emplace_back();
        for (const int c : local) {
            facets.push_back(same[static_cast<std::size_t>(c)]);
        }
        // The coarse triangle over each finest coordinate: the one it
        // is deepest inside. A polygon's sub-faces lie apart in its
        // parameterization, so its coordinates look for the nearest
        // in space instead.
        std::vector<std::array<int, 3>>& over =
            levels.over.emplace_back();
        for (int f = 0; f < levels.numCoords; ++f) {
            const float px =
                fineCoords[2 * static_cast<std::size_t>(f)];
            const float py =
                fineCoords[2 * static_cast<std::size_t>(f) + 1];
            float deepest = -std::numeric_limits<float>::infinity();
            std::array<int, 3> corners{-1, -1, -1};
            for (std::size_t t = 0;
                 t + 2 < local.size() && !param.HasSubFaces();
                 t += 3) {
                const float* a =
                    &coords[2 * static_cast<std::size_t>(local[t])];
                const float* b = &coords[2 * static_cast<std::size_t>(
                                                 local[t + 1])];
                const float* c = &coords[2 * static_cast<std::size_t>(
                                                 local[t + 2])];
                const float area = (b[0] - a[0]) * (c[1] - a[1]) -
                                   (b[1] - a[1]) * (c[0] - a[0]);
                if (std::abs(area) < 1e-12f) {
                    continue;
                }
                const float w1 = ((px - a[0]) * (c[1] - a[1]) -
                                  (py - a[1]) * (c[0] - a[0])) /
                                 area;
                const float w2 = ((b[0] - a[0]) * (py - a[1]) -
                                  (b[1] - a[1]) * (px - a[0])) /
                                 area;
                const float depth =
                    std::min({1.0f - w1 - w2, w1, w2});
                if (depth > deepest) {
                    deepest = depth;
                    corners = {facets[t], facets[t + 1],
                               facets[t + 2]};
                }
            }
            over.push_back(corners);
        }
    }
    return levels;
}

/// The squared distance from `p` to triangle `a`, `b`, `c`.
///
/// References:
///   C. Ericson, "Real-Time Collision Detection", Morgan Kaufmann,
///   2005, 5.1.5 (closest point on a triangle to a point).
float SquaredDistanceToTriangle(const float* p, const float* a,
                                const float* b, const float* c) {
    const auto sub = [](const float* x, const float* y) {
        return std::array<float, 3>{x[0] - y[0], x[1] - y[1],
                                    x[2] - y[2]};
    };
    const auto dot = [](const std::array<float, 3>& x,
                        const std::array<float, 3>& y) {
        return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    };
    const std::array<float, 3> ab = sub(b, a), ac = sub(c, a),
                               ap = sub(p, a);
    const auto squaredTo = [&](float v, float w) {
        float d = 0.0f;
        for (std::size_t k = 0; k < 3; ++k) {
            const float e = ap[k] - v * ab[k] - w * ac[k];
            d += e * e;
        }
        return d;
    };
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        return squaredTo(0.0f, 0.0f);
    }
    const std::array<float, 3> bp = sub(p, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        return squaredTo(1.0f, 0.0f);
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        return squaredTo(d1 / (d1 - d3), 0.0f);
    }
    const std::array<float, 3> cp = sub(p, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        return squaredTo(0.0f, 1.0f);
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        return squaredTo(0.0f, d2 / (d2 - d6));
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return squaredTo(1.0f - w, w);
    }
    const float denom = va + vb + vc;
    if (denom == 0.0f) {
        return squaredTo(0.0f, 0.0f);
    }
    return squaredTo(vb / denom, vc / denom);
}

/// Runs `work` on up to 32 threads, fewer for few `items`; each call
/// takes items itself. Rethrows the first exception.
template <typename Work>
void OnThreads(std::size_t items, const Work& work) {
    std::mutex errorMutex;
    std::exception_ptr error;
    const auto guarded = [&] {
        try {
            work();
        } catch (...) {
            const std::lock_guard lock(errorMutex);
            if (!error) {
                error = std::current_exception();
            }
        }
    };
    const std::size_t threads = std::clamp<std::size_t>(
        std::min<std::size_t>(std::thread::hardware_concurrency(),
                              items / 256 + 1),
        1, 32);
    std::vector<std::thread> pool;
    for (std::size_t i = 1; i < threads; ++i) {
        pool.emplace_back(guarded);
    }
    guarded();
    for (std::thread& thread : pool) {
        thread.join();
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

MeshData SubdivideWith(const SubdivisionSurface& surface, int level,
                       RefinementStencils* stencils);

}  // namespace

MeshData Subdivide(const SubdivisionSurface& surface, int level) {
    return SubdivideWith(surface, level, nullptr);
}

MeshData Subdivide(const SubdivisionSurface& surface, int level,
                   RefinementStencils& stencils) {
    return SubdivideWith(surface, level, &stencils);
}

namespace {

MeshData SubdivideWith(const SubdivisionSurface& surface, int level,
                       RefinementStencils* stencils) {
    if (surface.scheme == SubdivisionScheme::None) {
        throw std::invalid_argument(
            "A subdivision surface needs a scheme");
    }
    if (level < 0 || level > 8) {
        throw std::invalid_argument(
            "The subdivision level must be in [0, 8]");
    }
    if (surface.points.size() % 3 != 0) {
        throw std::invalid_argument("points must be 3-vectors");
    }
    const std::size_t pointCount = surface.points.size() / 3;
    if (!surface.colors.empty() &&
        surface.colors.size() != 3 * pointCount) {
        throw std::invalid_argument("colors need r, g, b per point");
    }
    const Topology topology = TopologyOf(surface);
    const std::unique_ptr<osd::Far::TopologyRefiner> refiner =
        topology.Refiner(surface, static_cast<int>(pointCount));
    const bool uvs = !topology.uvIndices.empty();
    const bool colors = !surface.colors.empty();
    using Factory = osd::Bfr::RefinerSurfaceFactory<>;
    Factory::Options factoryOptions;
    factoryOptions.SetDefaultFVarID(0);
    const osd::Sdc::SchemeType scheme = SchemeOf(surface.scheme);
    const int rate = 1 << level;
    const Layout layout = LayoutOf(
        *refiner, Factory(*refiner, factoryOptions), scheme, rate);

    MeshData data;
    data.points.assign(3 * layout.points, 0.0f);
    data.normals.assign(3 * layout.points, 0.0f);
    data.indices.assign(3 * layout.triangles, 0u);
    if (colors) {
        data.colors.assign(3 * layout.points, 0.0f);
    }
    if (uvs) {
        data.uvs.assign(6 * layout.triangles, 0.0f);
    }

    // The coarser levels, per face size, and where each face's
    // triangles go in each.
    const std::size_t faceCount = layout.faces.size();
    std::unordered_map<int, CoarseLevels> coarse;
    std::vector<const CoarseLevels*> faceLevels(faceCount, nullptr);
    std::vector<std::size_t> lodStart(
        static_cast<std::size_t>(level) * faceCount);
    if (level > 0) {
        const osd::Far::TopologyLevel& base = refiner->GetLevel(0);
        for (std::size_t f = 0; f < faceCount; ++f) {
            const int size =
                base.GetFaceVertices(layout.faces[f]).size();
            auto it = coarse.find(size);
            if (it == coarse.end()) {
                it =
                    coarse
                        .emplace(size, CoarseLevelsOf(
                                           osd::Bfr::Parameterization(
                                               scheme, size),
                                           rate))
                        .first;
            }
            faceLevels[f] = &it->second;
        }
        std::size_t total = 0;
        for (std::size_t l = 0; l < static_cast<std::size_t>(level);
             ++l) {
            const std::size_t first = total;
            for (std::size_t f = 0; f < faceCount; ++f) {
                lodStart[l * faceCount + f] = total;
                total += faceLevels[f]->facets[l].size();
            }
            if (total >= std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument(
                    "The subdivided mesh has too many triangles");
            }
            data.levels.push_back(
                {static_cast<std::uint32_t>(first),
                 static_cast<std::uint32_t>(total - first), 0.0f});
        }
        data.lodIndices.assign(total, 0u);
        if (uvs) {
            data.lodUvs.assign(2 * total, 0.0f);
        }
    }

    // Each point's stencil, written by the face that evaluates it.
    std::vector<std::vector<std::pair<std::uint32_t, float>>>
        pointStencils(stencils != nullptr ? layout.points : 0);

    // The faces evaluated in chunks on every thread, each with a
    // factory of its own (a factory's cache is not thread-safe).
    std::atomic<std::size_t> nextFace{0};
    OnThreads(faceCount, [&] {
        {
            const Factory factory(*refiner, factoryOptions);
            osd::Bfr::Surface<float> vertexSurface;
            osd::Bfr::Surface<float> uvSurface;
            std::vector<float> patch, colorPatch, uvPatch;
            std::vector<float> coords, uvAt, stencil;
            std::vector<int> facets, local, controls;
            osd::Bfr::Tessellation::Options options;
            options.SetFacetSize(3);
            constexpr std::size_t kChunk = 64;
            for (std::size_t begin = nextFace.fetch_add(kChunk);
                 begin < layout.faces.size();
                 begin = nextFace.fetch_add(kChunk)) {
                const std::size_t end =
                    std::min(layout.faces.size(), begin + kChunk);
                for (std::size_t f = begin; f < end; ++f) {
                    const int face = layout.faces[f];
                    if (!factory.InitVertexSurface(face,
                                                   &vertexSurface)) {
                        continue;
                    }
                    const osd::Bfr::Tessellation tess(
                        vertexSurface.GetParameterization(), rate,
                        options);
                    const int numCoords = tess.GetNumCoords();
                    const int numBoundary =
                        tess.GetNumBoundaryCoords();
                    coords.resize(
                        2 * static_cast<std::size_t>(numCoords));
                    tess.GetCoords(coords.data());
                    patch.resize(
                        3 * static_cast<std::size_t>(
                                vertexSurface.GetNumPatchPoints()));
                    vertexSurface.PreparePatchPoints(
                        surface.points.data(), 3, patch.data(), 3);
                    if (colors) {
                        colorPatch.resize(patch.size());
                        vertexSurface.PreparePatchPoints(
                            surface.colors.data(), 3,
                            colorPatch.data(), 3);
                    }
                    if (stencils != nullptr) {
                        controls.resize(static_cast<std::size_t>(
                            vertexSurface.GetNumControlPoints()));
                        vertexSurface.GetControlPointIndices(
                            controls.data());
                        stencil.resize(controls.size());
                    }
                    // Output point of each coordinate, and whether
                    // this face evaluates it.
                    const std::size_t b0 = layout.boundaryStart[f];
                    const auto pointOf = [&](int c) -> std::uint32_t {
                        return c < numBoundary
                                   ? layout.boundary
                                         [b0 +
                                          static_cast<std::size_t>(c)]
                                   : layout.interiorStart[f] +
                                         static_cast<std::uint32_t>(
                                             c - numBoundary);
                    };
                    for (int c = 0; c < numCoords; ++c) {
                        if (c < numBoundary &&
                            layout.owned[b0 +
                                         static_cast<std::size_t>(
                                             c)] == 0) {
                            continue;
                        }
                        const std::size_t p = 3 * pointOf(c);
                        float du[3];
                        float dv[3];
                        vertexSurface.Evaluate(
                            &coords[2 * static_cast<std::size_t>(c)],
                            patch.data(), 3, &data.points[p], du, dv);
                        const float n[3] = {
                            du[1] * dv[2] - du[2] * dv[1],
                            du[2] * dv[0] - du[0] * dv[2],
                            du[0] * dv[1] - du[1] * dv[0]};
                        const float length = std::sqrt(
                            n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                        if (length > 0.0f) {
                            for (int k = 0; k < 3; ++k) {
                                data.normals[p +
                                             static_cast<std::size_t>(
                                                 k)] = n[k] / length;
                            }
                        }
                        if (colors) {
                            vertexSurface.Evaluate(
                                &coords[2 *
                                        static_cast<std::size_t>(c)],
                                colorPatch.data(), 3,
                                &data.colors[p]);
                        }
                        if (stencils != nullptr) {
                            vertexSurface.EvaluateStencil(
                                &coords[2 *
                                        static_cast<std::size_t>(c)],
                                stencil.data());
                            auto& weights = pointStencils[pointOf(c)];
                            for (std::size_t k = 0;
                                 k < controls.size(); ++k) {
                                if (stencil[k] != 0.0f) {
                                    weights.emplace_back(
                                        static_cast<std::uint32_t>(
                                            controls[k]),
                                        stencil[k]);
                                }
                            }
                        }
                    }
                    const int numFacets = tess.GetNumFacets();
                    local.resize(3 *
                                 static_cast<std::size_t>(numFacets));
                    tess.GetFacets(local.data());
                    const std::size_t t0 =
                        3 * layout.triangleStart[f];
                    for (std::size_t k = 0; k < local.size(); ++k) {
                        data.indices[t0 + k] = pointOf(local[k]);
                    }
                    for (std::size_t l = 0; l < data.levels.size();
                         ++l) {
                        const std::vector<int>& coarseFacets =
                            faceLevels[f]->facets[l];
                        const std::size_t at =
                            lodStart[l * faceCount + f];
                        for (std::size_t k = 0;
                             k < coarseFacets.size(); ++k) {
                            data.lodIndices[at + k] =
                                pointOf(coarseFacets[k]);
                        }
                    }
                    if (uvs && factory.InitFaceVaryingSurface(
                                   face, &uvSurface)) {
                        uvPatch.resize(
                            2 * static_cast<std::size_t>(
                                    uvSurface.GetNumPatchPoints()));
                        uvSurface.PreparePatchPoints(
                            topology.uvValues.data(), 2,
                            uvPatch.data(), 2);
                        uvAt.resize(
                            2 * static_cast<std::size_t>(numCoords));
                        for (int c = 0; c < numCoords; ++c) {
                            const auto at =
                                2 * static_cast<std::size_t>(c);
                            uvSurface.Evaluate(&coords[at],
                                               uvPatch.data(), 2,
                                               &uvAt[at]);
                        }
                        for (std::size_t k = 0; k < local.size();
                             ++k) {
                            const auto at =
                                2 *
                                static_cast<std::size_t>(local[k]);
                            data.uvs[2 * (t0 + k)] = uvAt[at];
                            data.uvs[2 * (t0 + k) + 1] = uvAt[at + 1];
                        }
                        for (std::size_t l = 0;
                             l < data.levels.size(); ++l) {
                            const std::vector<int>& coarseFacets =
                                faceLevels[f]->facets[l];
                            const std::size_t first =
                                lodStart[l * faceCount + f];
                            for (std::size_t k = 0;
                                 k < coarseFacets.size(); ++k) {
                                const auto at =
                                    2 * static_cast<std::size_t>(
                                            coarseFacets[k]);
                                data.lodUvs[2 * (first + k)] =
                                    uvAt[at];
                                data.lodUvs[2 * (first + k) + 1] =
                                    uvAt[at + 1];
                            }
                        }
                    }
                }
            }
        }
    });

    // Where the derivatives vanish (some extraordinary points), the
    // normals of the triangles around the point.
    std::vector<std::uint32_t> missing;
    for (std::size_t p = 0; p < layout.points; ++p) {
        if (data.normals[3 * p] == 0.0f &&
            data.normals[3 * p + 1] == 0.0f &&
            data.normals[3 * p + 2] == 0.0f) {
            missing.push_back(static_cast<std::uint32_t>(p));
        }
    }
    if (!missing.empty()) {
        std::vector<float> sum(3 * layout.points, 0.0f);
        for (std::size_t t = 0; t < layout.triangles; ++t) {
            const std::uint32_t* tri = &data.indices[3 * t];
            const float* a =
                &data.points[3 * static_cast<std::size_t>(tri[0])];
            const float* b =
                &data.points[3 * static_cast<std::size_t>(tri[1])];
            const float* c =
                &data.points[3 * static_cast<std::size_t>(tri[2])];
            const float u[3] = {b[0] - a[0], b[1] - a[1],
                                b[2] - a[2]};
            const float v[3] = {c[0] - a[0], c[1] - a[1],
                                c[2] - a[2]};
            const float n[3] = {u[1] * v[2] - u[2] * v[1],
                                u[2] * v[0] - u[0] * v[2],
                                u[0] * v[1] - u[1] * v[0]};
            for (int corner = 0; corner < 3; ++corner) {
                for (int k = 0; k < 3; ++k) {
                    sum[3 * static_cast<std::size_t>(tri[corner]) +
                        static_cast<std::size_t>(k)] += n[k];
                }
            }
        }
        for (const std::uint32_t p : missing) {
            const float* n = &sum[3 * static_cast<std::size_t>(p)];
            const float length =
                std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (length > 0.0f) {
                for (int k = 0; k < 3; ++k) {
                    data.normals[3 * static_cast<std::size_t>(p) +
                                 static_cast<std::size_t>(k)] =
                        n[k] / length;
                }
            }
        }
    }

    // Each coarser level's error: the farthest a finest point lies
    // from that level's triangle over it.
    if (!data.levels.empty()) {
        std::mutex errorMutex;
        nextFace = 0;
        OnThreads(faceCount, [&] {
            std::vector<float> errors(data.levels.size(), 0.0f);
            constexpr std::size_t kChunk = 256;
            for (std::size_t begin = nextFace.fetch_add(kChunk);
                 begin < faceCount;
                 begin = nextFace.fetch_add(kChunk)) {
                const std::size_t end =
                    std::min(faceCount, begin + kChunk);
                for (std::size_t f = begin; f < end; ++f) {
                    const CoarseLevels& levels = *faceLevels[f];
                    const std::size_t b0 = layout.boundaryStart[f];
                    const auto pointAt = [&](int c) {
                        const std::uint32_t p =
                            c < levels.numBoundary
                                ? layout.boundary
                                      [b0 +
                                       static_cast<std::size_t>(c)]
                                : layout.interiorStart[f] +
                                      static_cast<std::uint32_t>(
                                          c - levels.numBoundary);
                        return &data.points[3 *
                                            static_cast<std::size_t>(
                                                p)];
                    };
                    for (std::size_t l = 0; l < data.levels.size();
                         ++l) {
                        for (int c = 0; c < levels.numCoords; ++c) {
                            const auto at =
                                static_cast<std::size_t>(c);
                            const std::array<int, 3>& corners =
                                levels.over[l][at];
                            const float* p = pointAt(c);
                            if (corners[0] >= 0) {
                                errors[l] = std::max(
                                    errors[l],
                                    SquaredDistanceToTriangle(
                                        p, pointAt(corners[0]),
                                        pointAt(corners[1]),
                                        pointAt(corners[2])));
                                continue;
                            }
                            const std::vector<int>& coarseFacets =
                                levels.facets[l];
                            float nearest = std::numeric_limits<
                                float>::infinity();
                            for (std::size_t t = 0;
                                 t + 2 < coarseFacets.size();
                                 t += 3) {
                                nearest = std::min(
                                    nearest,
                                    SquaredDistanceToTriangle(
                                        p, pointAt(coarseFacets[t]),
                                        pointAt(coarseFacets[t + 1]),
                                        pointAt(
                                            coarseFacets[t + 2])));
                            }
                            errors[l] = std::max(errors[l], nearest);
                        }
                    }
                }
            }
            const std::lock_guard lock(errorMutex);
            for (std::size_t l = 0; l < errors.size(); ++l) {
                data.levels[l].error = std::max(data.levels[l].error,
                                                std::sqrt(errors[l]));
            }
        });
    }
    if (stencils != nullptr) {
        stencils->starts.assign(1, 0u);
        stencils->indices.clear();
        stencils->weights.clear();
        for (const auto& weights : pointStencils) {
            for (const auto& [control, weight] : weights) {
                stencils->indices.push_back(control);
                stencils->weights.push_back(weight);
            }
            stencils->starts.push_back(
                static_cast<std::uint32_t>(stencils->indices.size()));
        }
    }
    return data;
}

}  // namespace
}  // namespace ernie

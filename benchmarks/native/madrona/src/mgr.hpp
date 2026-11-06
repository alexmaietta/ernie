#pragma once

#ifdef ernie_bench_madrona_mgr_EXPORTS
#define MGR_EXPORT MADRONA_EXPORT
#else
#define MGR_EXPORT MADRONA_IMPORT
#endif

#include <cstdint>
#include <memory>

#include <madrona/macros.hpp>

namespace ernie_bench {

struct Geometry {
    const float* vertices;
    const float* normals;
    const uint32_t* indices;
    const uint32_t* vertexOffsets;
    const uint32_t* triangleOffsets;
    uint32_t vertexCount;
    uint32_t triangleCount;
    uint32_t meshCount;
    const int32_t* objectMeshes;
    const float* objectColors;
    uint32_t objectCount;
};

class Manager {
  public:
    struct Config {
        int gpuID;
        uint32_t worldCount;
        uint32_t width;
        uint32_t height;
        bool useRT;
        float fovY;
        float nearPlane;
        float farPlane;
        float lightDirection[3];
    };

    MGR_EXPORT Manager(const Config& cfg, const Geometry& geometry, const int32_t* instance_objects,
                       uint32_t instance_count, uint32_t camera_count);
    MGR_EXPORT ~Manager();

    MGR_EXPORT void setInstances(const float* positions, const float* rotations,
                                 const float* scales);
    MGR_EXPORT void setCameras(const float* positions, const float* rotations);
    MGR_EXPORT void render();
    MGR_EXPORT void synchronize();
    MGR_EXPORT uintptr_t stream() const;
    MGR_EXPORT void readColor(uint8_t* out) const;
    MGR_EXPORT void readDepth(float* out) const;

    MGR_EXPORT uint32_t worldCount() const;
    MGR_EXPORT uint32_t instanceCount() const;
    MGR_EXPORT uint32_t cameraCount() const;
    MGR_EXPORT uint64_t pixelCount() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}

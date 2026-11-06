// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "ernie/device/compute/runtime_compiler.h"
#include "ernie/util/version.h"

namespace ernie::device::cuda {

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = uint64_t;
using CUcontext = struct CUctx_st*;
using CUstream = struct CUstream_st*;
using CUevent = struct CUevent_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUmemoryPool = struct CUmemPoolHandle_st*;

struct CUmemLocation {
    int type;
    int id;
};

struct CUmemPoolProps {
    int allocType;
    int handleTypes;
    CUmemLocation location;
    void* win32SecurityAttributes;
    size_t maxSize;
    unsigned char reserved[56];
};

using CUgraph = struct CUgraph_st*;
using CUgraphExec = struct CUgraphExec_st*;

struct CUuuid {
    unsigned char bytes[16];
};

using CUexternalSemaphore = struct CUextSemaphore_st*;
constexpr int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9;

struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
    int type;

    union {
        int fd;

        struct {
            void* handle;
            const void* name;
        } win32;

        const void* nvSciSyncObj;
    } handle;

    unsigned flags;
    unsigned reserved[16];
};

struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS {
    struct {
        struct {
            unsigned long long value;
        } fence;

        union {
            void* fence;
            unsigned long long reserved;
        } nvSciSync;

        struct {
            unsigned long long key;
        } keyedMutex;

        unsigned reserved[12];
    } params;

    unsigned flags;
    unsigned reserved[16];
};

struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
    struct {
        struct {
            unsigned long long value;
        } fence;

        union {
            void* fence;
            unsigned long long reserved;
        } nvSciSync;

        struct {
            unsigned long long key;
            unsigned timeoutMs;
        } keyedMutex;

        unsigned reserved[10];
    } params;

    unsigned flags;
    unsigned reserved[16];
};

using CUdevice_attribute = int;

using CUexternalMemory = struct CUextMemory_st*;
constexpr int CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1;
constexpr unsigned CUDA_EXTERNAL_MEMORY_DEDICATED = 0x1;

struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    int type;

    union {
        int fd;

        struct {
            void* handle;
            const void* name;
        } win32;

        const void* nvSciBufObject;
    } handle;

    unsigned long long size;
    unsigned flags;
    unsigned reserved[16];
};

static_assert(sizeof(CUDA_EXTERNAL_MEMORY_HANDLE_DESC) == 104);

struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
    unsigned long long offset;
    unsigned long long size;
    unsigned flags;
    unsigned reserved[16];
};

static_assert(sizeof(CUDA_EXTERNAL_MEMORY_BUFFER_DESC) == 88);

constexpr CUresult CUDA_SUCCESS = 0;
constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;
constexpr CUresult CUDA_ERROR_NO_DEVICE = 100;
constexpr CUresult CUDA_ERROR_NOT_FOUND = 500;
constexpr CUresult CUDA_ERROR_NOT_READY = 600;

constexpr unsigned CU_MEMHOSTALLOC_DEVICEMAP = 2;
constexpr unsigned CU_MEMHOSTALLOC_WRITECOMBINED = 4;
constexpr unsigned CU_STREAM_NON_BLOCKING = 1;
constexpr unsigned CU_EVENT_DISABLE_TIMING = 2;
constexpr int CU_MEM_ALLOCATION_TYPE_PINNED = 1;
constexpr int CU_MEM_LOCATION_TYPE_DEVICE = 1;
constexpr int CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4;
constexpr unsigned CU_EVENT_RECORD_EXTERNAL = 1;
constexpr int CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1;

constexpr int CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1;
constexpr int CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3;
constexpr int CU_FUNC_ATTRIBUTE_NUM_REGS = 4;

constexpr uintptr_t CU_LAUNCH_PARAM_END = 0;
constexpr uintptr_t CU_LAUNCH_PARAM_BUFFER_POINTER = 1;
constexpr uintptr_t CU_LAUNCH_PARAM_BUFFER_SIZE = 2;

constexpr bool LostDevice(CUresult result) {
    return result == 214 || result == 700 || result == 702 || (result >= 714 && result <= 719);
}

constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_INTEGRATED = 18;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115;

struct CUDA_ARRAY3D_DESCRIPTOR {
    size_t Width;
    size_t Height;
    size_t Depth;
    int Format;
    unsigned NumChannels;
    unsigned Flags;
};

struct CUDA_RESOURCE_DESC {
    int resType;

    union {
        struct {
            void* hMipmappedArray;
        } mipmap;

        int reserved[32];
    } res;

    unsigned flags;
};

struct CUDA_TEXTURE_DESC {
    int addressMode[3];
    int filterMode;
    unsigned flags;
    unsigned maxAnisotropy;
    int mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    float borderColor[4];
    int reserved[12];
};

struct CUDA_MEMCPY2D {
    size_t srcXInBytes;
    size_t srcY;
    int srcMemoryType;
    const void* srcHost;
    CUdeviceptr srcDevice;
    void* srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    int dstMemoryType;
    void* dstHost;
    CUdeviceptr dstDevice;
    void* dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
};

constexpr int CU_MEMORYTYPE_DEVICE = 2;
constexpr int CU_MEMORYTYPE_ARRAY = 3;

constexpr Version kMinimumDriverVersion{12, 0};

constexpr Version DecodeVersion(int packed) {
    return {packed / 1000, packed % 1000 / 10};
}

struct Api {
    std::string library;
    Version driverVersion;

    CUresult (*cuInit)(unsigned flags) = nullptr;
    CUresult (*cuDriverGetVersion)(int* version) = nullptr;
    CUresult (*cuDeviceGetCount)(int* count) = nullptr;
    CUresult (*cuGetErrorName)(CUresult result, const char** name) = nullptr;
    CUresult (*cuGetErrorString)(CUresult result, const char** text) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice* device, int ordinal) = nullptr;
    CUresult (*cuDeviceGetName)(char* name, int length, CUdevice device) = nullptr;
    CUresult (*cuDeviceGetUuid)(CUuuid* uuid, CUdevice device) = nullptr;
    CUresult (*cuDeviceTotalMem)(size_t* bytes, CUdevice device) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int* value, CUdevice_attribute attribute,
                                     CUdevice device) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext* context, CUdevice device) = nullptr;
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice device) = nullptr;
    CUresult (*cuCtxPushCurrent)(CUcontext context) = nullptr;
    CUresult (*cuCtxPopCurrent)(CUcontext* context) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr* pointer, size_t bytes) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr pointer) = nullptr;
    CUresult (*cuMemGetInfo)(size_t* free, size_t* total) = nullptr;
    CUresult (*cuMemPoolCreate)(CUmemoryPool* pool, const CUmemPoolProps* properties) = nullptr;
    CUresult (*cuMemPoolDestroy)(CUmemoryPool pool) = nullptr;
    CUresult (*cuMemPoolSetAttribute)(CUmemoryPool pool, int attribute, void* value) = nullptr;
    CUresult (*cuMemAllocFromPoolAsync)(CUdeviceptr* pointer, size_t bytes, CUmemoryPool pool,
                                        CUstream stream) = nullptr;
    CUresult (*cuMemFreeAsync)(CUdeviceptr pointer, CUstream stream) = nullptr;
    CUresult (*cuMemHostAlloc)(void** pointer, size_t bytes, unsigned flags) = nullptr;
    CUresult (*cuMemFreeHost)(void* pointer) = nullptr;
    CUresult (*cuMemHostGetDevicePointer)(CUdeviceptr* device, void* host,
                                          unsigned flags) = nullptr;
    CUresult (*cuStreamCreate)(CUstream* stream, unsigned flags) = nullptr;
    CUresult (*cuStreamDestroy)(CUstream stream) = nullptr;
    CUresult (*cuStreamSynchronize)(CUstream stream) = nullptr;
    CUresult (*cuEventCreate)(CUevent* event, unsigned flags) = nullptr;
    CUresult (*cuEventDestroy)(CUevent event) = nullptr;
    CUresult (*cuEventQuery)(CUevent event) = nullptr;
    CUresult (*cuEventSynchronize)(CUevent event) = nullptr;
    CUresult (*cuEventRecordWithFlags)(CUevent event, CUstream stream, unsigned flags) = nullptr;
    CUresult (*cuStreamWaitEvent)(CUstream stream, CUevent event, unsigned flags) = nullptr;
    CUresult (*cuLaunchHostFunc)(CUstream stream, void (*function)(void*), void* data) = nullptr;
    CUresult (*cuImportExternalSemaphore)(
        CUexternalSemaphore* semaphore, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* desc) = nullptr;
    CUresult (*cuDestroyExternalSemaphore)(CUexternalSemaphore semaphore) = nullptr;
    CUresult (*cuImportExternalMemory)(CUexternalMemory* memory,
                                       const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* desc) = nullptr;
    CUresult (*cuExternalMemoryGetMappedBuffer)(CUdeviceptr* pointer, CUexternalMemory memory,
                                                const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* desc) =
        nullptr;
    CUresult (*cuDestroyExternalMemory)(CUexternalMemory memory) = nullptr;
    CUresult (*cuMemcpyDtoH)(void* dst, CUdeviceptr src, size_t bytes) = nullptr;
    CUresult (*cuSignalExternalSemaphoresAsync)(const CUexternalSemaphore* semaphores,
                                                const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* params,
                                                unsigned count, CUstream stream) = nullptr;
    CUresult (*cuWaitExternalSemaphoresAsync)(const CUexternalSemaphore* semaphores,
                                              const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* params,
                                              unsigned count, CUstream stream) = nullptr;
    CUresult (*cuEventElapsedTime)(float* milliseconds, CUevent from, CUevent to) = nullptr;
    CUresult (*cuMemcpyAsync)(CUdeviceptr dst, CUdeviceptr src, size_t bytes,
                              CUstream stream) = nullptr;
    CUresult (*cuMemsetD32Async)(CUdeviceptr dst, unsigned value, size_t count,
                                 CUstream stream) = nullptr;
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr dst, const void* src, size_t bytes,
                                  CUstream stream) = nullptr;
    CUresult (*cuModuleLoadData)(CUmodule* module, const void* image) = nullptr;
    CUresult (*cuModuleUnload)(CUmodule module) = nullptr;
    CUresult (*cuModuleGetGlobal)(CUdeviceptr* pointer, size_t* bytes, CUmodule module,
                                  const char* name) = nullptr;
    CUresult (*cuMemcpyHtoD)(CUdeviceptr dst, const void* src, size_t bytes) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction* function, CUmodule module,
                                    const char* name) = nullptr;
    CUresult (*cuFuncGetAttribute)(int* value, int attribute, CUfunction function) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction function, unsigned groupsX, unsigned groupsY,
                               unsigned groupsZ, unsigned threadsX, unsigned threadsY,
                               unsigned threadsZ, unsigned sharedBytes, CUstream stream,
                               void** params, void** extra) = nullptr;
    CUresult (*cuStreamBeginCapture)(CUstream stream, int mode) = nullptr;
    CUresult (*cuStreamEndCapture)(CUstream stream, CUgraph* graph) = nullptr;
    CUresult (*cuGraphInstantiateWithFlags)(CUgraphExec* exec, CUgraph graph,
                                            unsigned long long flags) = nullptr;
    CUresult (*cuGraphLaunch)(CUgraphExec exec, CUstream stream) = nullptr;
    CUresult (*cuGraphExecDestroy)(CUgraphExec exec) = nullptr;
    CUresult (*cuGraphDestroy)(CUgraph graph) = nullptr;
    CUresult (*cuMipmappedArrayCreate)(void** array, const CUDA_ARRAY3D_DESCRIPTOR* desc,
                                       unsigned levels) = nullptr;
    CUresult (*cuMipmappedArrayGetLevel)(void** level, void* array, unsigned index) = nullptr;
    CUresult (*cuMipmappedArrayDestroy)(void* array) = nullptr;
    CUresult (*cuMemcpy2DAsync)(const CUDA_MEMCPY2D* copy, CUstream stream) = nullptr;
    CUresult (*cuTexObjectCreate)(unsigned long long* object, const CUDA_RESOURCE_DESC* resource,
                                  const CUDA_TEXTURE_DESC* sampler, const void* view) = nullptr;
    CUresult (*cuTexObjectDestroy)(unsigned long long object) = nullptr;

    const char* ErrorName(CUresult result) const;

    void Warn(CUresult result, const char* what) const;

    void Check(CUresult result, const char* what) const;
};

bool Available();

const Api& GetApi();

const compute::RuntimeCompiler& GetNvrtc();

}

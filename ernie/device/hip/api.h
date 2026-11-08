// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "ernie/device/compute/runtime_compiler.h"
#include "ernie/util/version.h"

namespace ernie::device::hip {

using hipError_t = int;
using hipDevice_t = int;
using hipDeviceptr_t = void*;
using hipStream_t = struct ihipStream_t*;
using hipEvent_t = struct ihipEvent_t*;
using hipModule_t = struct ihipModule_t*;
using hipFunction_t = struct ihipModuleSymbol_t*;
using hipMemPool_t = struct ihipMemPoolHandle_t*;

struct hipMemLocation {
    int type;
    int id;
};

struct hipMemPoolProps {
    int allocType;
    int handleTypes;
    hipMemLocation location;
    void* win32SecurityAttributes;
    size_t maxSize;
    unsigned char reserved[56];
};

using hipGraph_t = struct ihipGraph*;
using hipGraphExec_t = struct hipGraphExec*;

struct hipUUID {
    char bytes[16];
};

using hipExternalSemaphore_t = void*;
constexpr int hipExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9;

struct hipExternalSemaphoreHandleDesc {
    int type;

    union {
        int fd;

        struct {
            void* handle;
            const void* name;
        } win32;

        const void* NvSciSyncObj;
    } handle;

    unsigned flags;
    unsigned reserved[16];
};

struct hipExternalSemaphoreSignalParams {
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

struct hipExternalSemaphoreWaitParams {
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

using hipDeviceAttribute_t = int;

using hipExternalMemory_t = void*;
constexpr int hipExternalMemoryHandleTypeOpaqueFd = 1;
constexpr unsigned hipExternalMemoryDedicated = 0x1;

struct hipExternalMemoryHandleDesc {
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

static_assert(sizeof(hipExternalMemoryHandleDesc) == 104);

struct hipExternalMemoryBufferDesc {
    unsigned long long offset;
    unsigned long long size;
    unsigned flags;
    unsigned reserved[16];
};

static_assert(sizeof(hipExternalMemoryBufferDesc) == 88);

constexpr hipError_t hipSuccess = 0;
constexpr hipError_t hipErrorOutOfMemory = 2;
constexpr hipError_t hipErrorNoDevice = 100;
constexpr hipError_t hipErrorNotFound = 500;
constexpr hipError_t hipErrorNotReady = 600;

constexpr unsigned hipHostMallocMapped = 2;
constexpr unsigned hipHostMallocWriteCombined = 4;
constexpr unsigned hipStreamNonBlocking = 1;
constexpr unsigned hipEventDisableTiming = 2;
constexpr int hipMemAllocationTypePinned = 1;
constexpr int hipMemLocationTypeDevice = 1;
constexpr int hipMemPoolAttrReleaseThreshold = 4;
constexpr unsigned hipEventRecordExternal = 1;
constexpr int hipStreamCaptureModeThreadLocal = 1;
constexpr int hipMemcpyDefault = 4;

constexpr int HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1;
constexpr int HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3;
constexpr int HIP_FUNC_ATTRIBUTE_NUM_REGS = 4;

constexpr uintptr_t HIP_LAUNCH_PARAM_BUFFER_POINTER = 1;
constexpr uintptr_t HIP_LAUNCH_PARAM_BUFFER_SIZE = 2;
constexpr uintptr_t HIP_LAUNCH_PARAM_END = 3;

constexpr bool LostDevice(hipError_t error) {
    return error == 214 || error == 700 || error == 702 || (error >= 714 && error <= 719);
}

constexpr hipDeviceAttribute_t hipDeviceAttributeIntegrated = 16;
constexpr hipDeviceAttribute_t hipDeviceAttributeMaxThreadsPerBlock = 56;
constexpr hipDeviceAttribute_t hipDeviceAttributeMultiprocessorCount = 63;
constexpr hipDeviceAttribute_t hipDeviceAttributeWarpSize = 87;
constexpr hipDeviceAttribute_t hipDeviceAttributeMemoryPoolsSupported = 88;
constexpr hipDeviceAttribute_t hipDeviceAttributeImageSupport = 10014;

constexpr size_t kDevicePropertiesR0600Bytes = 1472;
constexpr size_t kDevicePropertiesR0600ArchNameOffset = 1160;

constexpr Version kMinimumRuntimeVersion{6, 2};

constexpr Version DecodeVersion(int packed) {
    return {packed / 10000000, packed / 100000 % 100, packed % 100000};
}

struct HIP_ARRAY3D_DESCRIPTOR {
    size_t Width;
    size_t Height;
    size_t Depth;
    int Format;
    unsigned NumChannels;
    unsigned Flags;
};

struct HIP_RESOURCE_DESC {
    int resType;

    union {
        struct {
            void* hMipmappedArray;
        } mipmap;

        int reserved[32];
    } res;

    unsigned flags;
};

struct HIP_TEXTURE_DESC {
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

struct hip_Memcpy2D {
    size_t srcXInBytes;
    size_t srcY;
    int srcMemoryType;
    const void* srcHost;
    hipDeviceptr_t srcDevice;
    void* srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    int dstMemoryType;
    void* dstHost;
    hipDeviceptr_t dstDevice;
    void* dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
};

constexpr int hipMemoryTypeDevice = 2;
constexpr int hipMemoryTypeArray = 10;

struct Api {
    std::string library;
    Version runtimeVersion;
    Version driverVersion;

    hipError_t (*hipInit)(unsigned flags) = nullptr;
    hipError_t (*hipRuntimeGetVersion)(int* version) = nullptr;
    hipError_t (*hipDriverGetVersion)(int* version) = nullptr;
    hipError_t (*hipGetDeviceCount)(int* count) = nullptr;
    const char* (*hipGetErrorName)(hipError_t error) = nullptr;
    const char* (*hipGetErrorString)(hipError_t error) = nullptr;
    hipError_t (*hipGetLastError)() = nullptr;
    hipError_t (*hipPeekAtLastError)() = nullptr;
    hipError_t (*hipDeviceGetName)(char* name, int length, hipDevice_t device) = nullptr;
    hipError_t (*hipDeviceGetUuid)(hipUUID* uuid, hipDevice_t device) = nullptr;
    hipError_t (*hipDeviceTotalMem)(size_t* bytes, hipDevice_t device) = nullptr;
    hipError_t (*hipDeviceGetAttribute)(int* value, hipDeviceAttribute_t attribute,
                                        int device) = nullptr;
    hipError_t (*hipGetDevicePropertiesR0600)(void* properties, int device) = nullptr;
    hipError_t (*hipGetDevice)(int* device) = nullptr;
    hipError_t (*hipSetDevice)(int device) = nullptr;
    hipError_t (*hipMalloc)(void** pointer, size_t bytes) = nullptr;
    hipError_t (*hipFree)(void* pointer) = nullptr;
    hipError_t (*hipMemGetInfo)(size_t* free, size_t* total) = nullptr;
    hipError_t (*hipMemPoolCreate)(hipMemPool_t* pool, const hipMemPoolProps* properties) = nullptr;
    hipError_t (*hipMemPoolDestroy)(hipMemPool_t pool) = nullptr;
    hipError_t (*hipMemPoolSetAttribute)(hipMemPool_t pool, int attribute, void* value) = nullptr;
    hipError_t (*hipMallocFromPoolAsync)(void** pointer, size_t bytes, hipMemPool_t pool,
                                         hipStream_t stream) = nullptr;
    hipError_t (*hipFreeAsync)(void* pointer, hipStream_t stream) = nullptr;
    hipError_t (*hipHostMalloc)(void** pointer, size_t bytes, unsigned flags) = nullptr;
    hipError_t (*hipHostFree)(void* pointer) = nullptr;
    hipError_t (*hipHostGetDevicePointer)(void** device, void* host, unsigned flags) = nullptr;
    hipError_t (*hipStreamCreateWithFlags)(hipStream_t* stream, unsigned flags) = nullptr;
    hipError_t (*hipStreamDestroy)(hipStream_t stream) = nullptr;
    hipError_t (*hipStreamSynchronize)(hipStream_t stream) = nullptr;
    hipError_t (*hipEventCreateWithFlags)(hipEvent_t* event, unsigned flags) = nullptr;
    hipError_t (*hipEventDestroy)(hipEvent_t event) = nullptr;
    hipError_t (*hipEventQuery)(hipEvent_t event) = nullptr;
    hipError_t (*hipEventSynchronize)(hipEvent_t event) = nullptr;
    hipError_t (*hipEventRecordWithFlags)(hipEvent_t event, hipStream_t stream,
                                          unsigned flags) = nullptr;
    hipError_t (*hipStreamWaitEvent)(hipStream_t stream, hipEvent_t event,
                                     unsigned flags) = nullptr;
    hipError_t (*hipLaunchHostFunc)(hipStream_t stream, void (*function)(void*),
                                    void* data) = nullptr;
    hipError_t (*hipImportExternalSemaphore)(hipExternalSemaphore_t* semaphore,
                                             const hipExternalSemaphoreHandleDesc* desc) = nullptr;
    hipError_t (*hipDestroyExternalSemaphore)(hipExternalSemaphore_t semaphore) = nullptr;
    hipError_t (*hipImportExternalMemory)(hipExternalMemory_t* memory,
                                          const hipExternalMemoryHandleDesc* desc) = nullptr;
    hipError_t (*hipExternalMemoryGetMappedBuffer)(void** pointer, hipExternalMemory_t memory,
                                                   const hipExternalMemoryBufferDesc* desc) =
        nullptr;
    hipError_t (*hipDestroyExternalMemory)(hipExternalMemory_t memory) = nullptr;
    hipError_t (*hipMemcpyDtoH)(void* dst, hipDeviceptr_t src, size_t bytes) = nullptr;
    hipError_t (*hipSignalExternalSemaphoresAsync)(const hipExternalSemaphore_t* semaphores,
                                                   const hipExternalSemaphoreSignalParams* params,
                                                   unsigned count, hipStream_t stream) = nullptr;
    hipError_t (*hipWaitExternalSemaphoresAsync)(const hipExternalSemaphore_t* semaphores,
                                                 const hipExternalSemaphoreWaitParams* params,
                                                 unsigned count, hipStream_t stream) = nullptr;
    hipError_t (*hipEventElapsedTime)(float* milliseconds, hipEvent_t from,
                                      hipEvent_t to) = nullptr;
    hipError_t (*hipMemcpyAsync)(void* dst, const void* src, size_t bytes, int kind,
                                 hipStream_t stream) = nullptr;
    hipError_t (*hipMemsetD32Async)(hipDeviceptr_t dst, int value, size_t count,
                                    hipStream_t stream) = nullptr;
    hipError_t (*hipMemcpyHtoDAsync)(hipDeviceptr_t dst, const void* src, size_t bytes,
                                     hipStream_t stream) = nullptr;
    hipError_t (*hipModuleLoadData)(hipModule_t* module, const void* image) = nullptr;
    hipError_t (*hipModuleUnload)(hipModule_t module) = nullptr;
    hipError_t (*hipModuleGetGlobal)(hipDeviceptr_t* pointer, size_t* bytes, hipModule_t module,
                                     const char* name) = nullptr;
    hipError_t (*hipMemcpyHtoD)(hipDeviceptr_t dst, const void* src, size_t bytes) = nullptr;
    hipError_t (*hipModuleGetFunction)(hipFunction_t* function, hipModule_t module,
                                       const char* name) = nullptr;
    hipError_t (*hipFuncGetAttribute)(int* value, int attribute, hipFunction_t function) = nullptr;
    hipError_t (*hipModuleLaunchKernel)(hipFunction_t function, unsigned groupsX, unsigned groupsY,
                                        unsigned groupsZ, unsigned threadsX, unsigned threadsY,
                                        unsigned threadsZ, unsigned sharedBytes, hipStream_t stream,
                                        void** params, void** extra) = nullptr;
    hipError_t (*hipStreamBeginCapture)(hipStream_t stream, int mode) = nullptr;
    hipError_t (*hipStreamEndCapture)(hipStream_t stream, hipGraph_t* graph) = nullptr;
    hipError_t (*hipGraphInstantiateWithFlags)(hipGraphExec_t* exec, hipGraph_t graph,
                                               unsigned long long flags) = nullptr;
    hipError_t (*hipGraphLaunch)(hipGraphExec_t exec, hipStream_t stream) = nullptr;
    hipError_t (*hipGraphExecDestroy)(hipGraphExec_t exec) = nullptr;
    hipError_t (*hipGraphDestroy)(hipGraph_t graph) = nullptr;
    hipError_t (*hipMipmappedArrayCreate)(void** array, const HIP_ARRAY3D_DESCRIPTOR* desc,
                                          unsigned levels) = nullptr;
    hipError_t (*hipMipmappedArrayGetLevel)(void** level, void* array, unsigned index) = nullptr;
    hipError_t (*hipMipmappedArrayDestroy)(void* array) = nullptr;
    hipError_t (*hipMemcpyParam2DAsync)(const hip_Memcpy2D* copy, hipStream_t stream) = nullptr;
    hipError_t (*hipTexObjectCreate)(void** object, const HIP_RESOURCE_DESC* resource,
                                     const HIP_TEXTURE_DESC* sampler, const void* view) = nullptr;
    hipError_t (*hipTexObjectDestroy)(void* object) = nullptr;

    const char* ErrorName(hipError_t error) const;

    void Warn(hipError_t error, const char* what) const;

    void Check(hipError_t error, const char* what) const;
};

bool Available();

const Api& GetApi();

const compute::RuntimeCompiler& GetHiprtc();

}

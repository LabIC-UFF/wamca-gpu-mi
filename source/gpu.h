/**
 * @file	gpu.h
 *
 * @brief	CUDA functions.
 *
 * @author	Eyder Rios
 * @date	2015-12-05
 */


#ifndef __cuda_gpu_h
#define __cuda_gpu_h

#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda/gpu_helper.h>
#include <cuda/gpu_string.h>
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>


#define KERNEL      __global__
#define DEVICE      __device__
#define HOST        __host__

/*!
 * Size of ushort type in bits
 */
#define BITS_USHORT         (sizeof(ushort) * CHAR_BIT)
/*!
 * Size of uint type in bits
 */
#define BITS_UINT           (sizeof(uint)   * CHAR_BIT)
/*!
 * Size of ullong type in bits
 */
#define BITS_ULLONG         (sizeof(ullong) * CHAR_BIT)
/*!
 * Shared memory aligment
 */
#define GPU_SHARED_ALIGN    sizeof(int)
/*!
 * Global memory aligment
 */
#define GPU_GLOBAL_ALIGN    128

#define GPU_BLKSIZE_CEIL(n,b)       (GPU_DIVCEIL(n,b) * (b))
#define GPU_BLKSIZE_GLOBAL(n)       GPU_BLKSIZE_CEIL(n,GPU_GLOBAL_ALIGN)
#define GPU_BLKSIZE_SHARED(n)       GPU_BLKSIZE_CEIL(n,GPU_SHARED_ALIGN)

/*!
 * Get data size in bits
 */
#define bitsize(t)              (sizeof(t) * CHAR_BIT)
/*!
 * Store two ushort in a single uint
 */
#define GPU_USHORT2UINT(h,l)    ( (uint(ushort(h)) << BITS_USHORT) | ushort(l) )
#define GPU_LO_USHORT(w)        ushort(w)
#define GPU_HI_USHORT(w)        ushort((w) >> BITS_USHORT)
/*!
 * Store two uint in a single ulong
 */
#define GPU_UINT2ULONG(h,l)     ( (ulong(uint(h)) << BITS_UINT) | uint(l) )
#define GPU_LO_UINT(w)          uint(w)
#define GPU_HI_UINT(w)          uint((w) >> BITS_UINT)

/*!
 * Square of an value
 */
#define GPU_SQR(x)              ( (x)*(x) )
/*!
 * Mask for bitwse calculations.
 */
#define GPU_MASK(x)             ( (x) >> sizeof(int)*CHAR_BIT - 1 )
/*!
 * Absolute value.
 */
#define GPU_ABS(x)              ( ( (x) + GPU_MASK(x) ) ^ GPU_MASK(x) )
/*!
 * Minimum between two integers.
 */
#define GPU_MIN(x,y)            ( (y) + ( ((x) - (y)) & GPU_MASK((x) - (y)) ) )
/*!
 * Maximum between two integers.
 */
#define GPU_MAX(x,y)            ( (x) - ( ((x) - (y)) & GPU_MASK((x) - (y)) ) )
/*!
 * Signal of an integer.
 *  x < 0  -1
 *  x = 0   0
 *  x > 0  +1
 */
#define GPU_SIGN(x)             ( ( (x) > 0) - ( (x) < 0) )
/*!
 * Calculate the number of words of type 't' to store 'n' bits
 */
#define GPU_BIT2WORD(n,t)       ( ((n) / (sizeof(t)*CHAR_BIT)) + (((n) % (sizeof(t)*CHAR_BIT)) != 0) )
/*!
 * Calculate the number of bytes to store 'n' bits
 */
#define GPU_BIT2BYTE(n)         ( ((n) / CHAR_BIT) + (((n) % CHAR_BIT) != 0) )
/*!
 * Ceil of integer division
 */
#define GPU_DIVCEIL(n,d)        ( (n + d - 1) / (d) )
/*!
 * Floor of integer division
 */
#define GPU_DIVFLOOR(n,d)       ( (n) / (d) )

/*!
 * Block and address alignment
 */
#define GPU_ALIGN_SIZE          sizeof(int)

#define GPU_BLOCK_SIZE(sz)      ( (((sz) / GPU_ALIGN_SIZE) + ((sz) % GPU_ALIGN_SIZE != 0)) * GPU_ALIGN_SIZE )
#define GPU_BLOCK_CEIL(t,sz)    GPU_BLOCK_SIZE(sizeof(t) * (sz))
#define GPU_BLOCK_CAST(v,p,sz)  (v) = (typeof(*v) *)( p )
#define GPU_BLOCK_NEXT(v,p,sz)  (v) = ( (typeof(*v) *)( ((byte *) (p)) + GPU_BLOCK_CEIL(*p,sz) ) )

/*!
 * ADS TIME
 */
#define GPU_ADS_TIME(i,j)       (sm_time[j] - sm_time[i])


#if 0
/*!
 * Element box coordinates (by solution index)
 */
#define GPU_COORD_X1(sol,i)         ( s_coordx[i] + s_dx1[ sol[i] ] )
#define GPU_COORD_X2(sol,i)         ( s_coordx[i] + s_dx2[ sol[i] ] )
#define GPU_COORD_Y1(sol,i)         ( s_coordy[i] + s_dy1[ sol[i] ] )
#define GPU_COORD_Y2(sol,i)         ( s_coordy[i] + s_dy2[ sol[i] ] )
/*!
 * Element box coordinates (by candidate position)
 */
#define GPU_SOLX_X1(i)             ( s_coordx[i] + s_dx1[ s_solbase[i] ] )
#define GPU_SOLX_X2(i)             ( s_coordx[i] + s_dx2[ s_solbase[i] ] )
#define GPU_SOLX_Y1(i)             ( s_coordy[i] + s_dy1[ s_solbase[i] ] )
#define GPU_SOLX_Y2(i)             ( s_coordy[i] + s_dy2[ s_solbase[i] ] )
/*!
 * Element box coordinates (by candidate position)
 */
#define GPU_CAND_X1(i,cp)           ( s_coordx[i] + s_dx1[cp] )
#define GPU_CAND_X2(i,cp)           ( s_coordx[i] + s_dx2[cp] )
#define GPU_CAND_Y1(i,cp)           ( s_coordy[i] + s_dy1[cp] )
#define GPU_CAND_Y2(i,cp)           ( s_coordy[i] + s_dy2[cp] )
#endif

#define GPU_STR(s)      #s
#define GPU_STR_VAL(s)  GPU_STR(s)

//#undef  GPU_CUDA_DISABLED
//#define GPU_CUDA_TRACE

#ifndef GPU_CUDA_DISABLED
/*
 * CUDA ENABLED
 */
#ifdef  GPU_CUDA_TRACE
#define GPU_CHECK_ERROR(fnc)            { lprintf("%s\n",GPU_STR_VAL(fnc)); checkCudaErrors(fnc); }
#else
#define GPU_CHECK_ERROR(fnc)            checkCudaErrors(fnc)
#endif

#define gpuMemGetInfo(f,t)              GPU_CHECK_ERROR(cudaMemGetInfo(f,t))

#define gpuMalloc(p,sz)                 GPU_CHECK_ERROR(cudaMalloc((void **) (p),sz))
#define gpuFree(p)                      GPU_CHECK_ERROR(cudaFree(p))

#define gpuHostMalloc(p,s,f)            GPU_CHECK_ERROR(__cudaHostMalloc((void **) (p),s,f))
#define gpuHostFree(p,f)                GPU_CHECK_ERROR(__cudaHostFree(p,f))

#define gpuOccupancyMaxPotentialBlockSize(g,b,f,l)                  cudaOccupancyMaxPotentialBlockSize(g,b,f,l)
#define gpuOccupancyMaxPotentialBlockSizeVariableSMem(g,b,f,s,l)    cudaOccupancyMaxPotentialBlockSizeVariableSMem(g,b,f,s,l)

#else
/*
 * CUDA DISABLED
 */
#define GPU_CHECK_ERROR(fnc)

#define gpuMemGetInfo(f,t)              *f = *t = 0

#define gpuMalloc(p,sz)                 __cudaHostMalloc((void **) (p),sz,cudaHostAllocPaged)
#define gpuFree(p)                      __cudaHostFree(p,cudaHostAllocPaged)

#define gpuHostMalloc(p,s,f)            __cudaHostMalloc((void **) (p),s,cudaHostAllocPaged)
#define gpuHostFree(p,f)                __cudaHostFree(p,cudaHostAllocPaged)

#define gpuOccupancyMaxPotentialBlockSize(g,b,f,l)                  cudaSuccess
#define gpuOccupancyMaxPotentialBlockSizeVariableSMem(g,b,f,s,l)    cudaSuccess

#endif

#define gpuSetDevice(id)                GPU_CHECK_ERROR(cudaSetDevice(id))
#define gpuGetDevice(id)                GPU_CHECK_ERROR(cudaGetDevice(id))
#define gpuDeviceReset()                GPU_CHECK_ERROR(cudaDeviceReset())

#define gpuMemset(d,v,sz)               GPU_CHECK_ERROR(cudaMemset(d,v,sz))
#define gpuMemcpy(d,s,sz,k)             GPU_CHECK_ERROR(cudaMemcpy(d,s,sz,k))
#define gpuMemcpyAsync(d,s,sz,k,f)      GPU_CHECK_ERROR(cudaMemcpyAsync(d,s,sz,k,f))
#define gpuGetDeviceCount(c)            GPU_CHECK_ERROR(cudaGetDeviceCount(c))
#define gpuGetDeviceProperties(p,d)     GPU_CHECK_ERROR(cudaGetDeviceProperties(p,d))
#define gpuDeviceSynchronize()          GPU_CHECK_ERROR(cudaDeviceSynchronize())

#define gpuEventCreate(e)               GPU_CHECK_ERROR(cudaEventCreate(e))
#define gpuEventDestroy(e)              GPU_CHECK_ERROR(cudaEventDestroy(e))
#define gpuEventRecord(e,s)             GPU_CHECK_ERROR(cudaEventRecord(e,s))
#define gpuEventSynchronize(e)          GPU_CHECK_ERROR(cudaEventSynchronize(e))
#define gpuEventElapsedTime(m,s,e)      GPU_CHECK_ERROR(cudaEventElapsedTime(m,s,e))

#define gpuStreamCreate(s)              GPU_CHECK_ERROR(cudaStreamCreate(s))
#define gpuStreamDestroy(s)             GPU_CHECK_ERROR(cudaStreamDestroy(s))
#define gpuStreamSynchronize(s)         GPU_CHECK_ERROR(cudaStreamSynchronize(s))

#define gpuStreamAddCallback(s,c,p,f)   GPU_CHECK_ERROR(cudaStreamAddCallback(s,c,p,f))

#define gpuProfilerStart()              GPU_CHECK_ERROR(cudaProfilerStart())
#define gpuProfilerStop()               GPU_CHECK_ERROR(cudaProfilerStop())

#define gpuDriverGetVersion(v)          GPU_CHECK_ERROR(cudaDriverGetVersion(v))
#define gpuRuntimeGetVersion(v)         GPU_CHECK_ERROR(cudaRuntimeGetVersion(v))

/*!
 * Flag for make cudaHostAlloc() performs like malloc()
 */
#define cudaHostAllocPaged      0xff

/*!
 * Allocates memory on the host.
 *
 * The flags parameter enables different options to be specified that affect the allocation,
 * as follows.
 *
 * cudaHostAllocPaged           This flags allocates paged-memory, just like malloc().
 *
 * cudaHostAllocDefault         This flag's value is defined to be 0 and causes cudaHostAlloc()
 *                              to emulate cudaMallocHost().
 * cudaHostAllocPortable        The memory returned by this call will be considered as pinned memory
 *                              by all CUDA contexts, not just the one that performed the allocation.
 * cudaHostAllocMapped          Maps the allocation into the CUDA address space. The device pointer
 *                              to the memory may be obtained by calling cudaHostGetDevicePointer().
 * cudaHostAllocWriteCombined   Allocates the memory as write-combined (WC). WC memory can be
 *                              transferred across the PCI Express bus more quickly on some system
 *                              configurations, but cannot be read efficiently by most CPUs.
 *                              WC memory is a good option for buffers that will be written by the
 *                              CPU and read by the device via mapped pinned memory or
 *                              host->device transfers.
 *
 * @param   p       Device pointer to allocated memory
 * @param   size    Requested allocation size in bytes
 * @params  flags   Requested properties of allocated memory
 *
 * @return  Returns error code.
 */
inline
cudaError_t
__cudaHostMalloc(void **p, size_t size, int flags) {
    cudaError_t error;
#ifndef GPU_CUDA_DISABLED
    if(flags == cudaHostAllocPaged) {
        if((*p = malloc(size)) == NULL)
            error = cudaErrorMemoryAllocation;
        else
            error = cudaSuccess;
    }
    else {
        *p = NULL;
        error = cudaHostAlloc(p,size,flags);
    }
#else
    if((*p = malloc(size)) == NULL)
        error = cudaErrorMemoryAllocation;
    else
        error = cudaSuccess;
#endif
    //printf("%s(%p,%ld,%d)\n",__FUNCTION__,p,size,flags);
    return error;
}
/*!
 * Free dynamic allocated memory on the host.
 *
 * The flags parameter enables different options to be specified that affect the memory releasing.
 * If \a flag equal to \a cudaHostAllocPagged, function peforms just like free(), otherwise
 * performs like cudaFreeHost().
 *
 * @param   p       Device pointer to allocated memory
 *
 * @return  Returns error code.
 */
inline
cudaError_t
__cudaHostFree(void *p, int flags) {
    cudaError_t error;
#ifndef GPU_CUDA_DISABLED
    if(flags == cudaHostAllocPaged) {
        free(p);
        error = cudaSuccess;
    }
    else
        error = cudaFreeHost(p);
#else
    if(p) {
        free(p);
        error = cudaSuccess;
    }
    else
        error = cudaErrorInitializationError;
#endif
    //printf("%s(%p,%d)\n",__FUNCTION__,p,flags);
    return error;
}

#if 0

#undef  gpuMalloc
#undef  gpuFree

#define gpuMalloc(p,s)          { int gid; cudaGetDevice(&gid); \
                                  fprintf(stderr,"%-15.15s: call gpuMalloc(%p,%d) <",__FUNCTION__,p,s); \
                                  checkCudaErrors(cudaMalloc((void **) (p),s)); fprintf(stderr,"*p=%p>\n",*p); }
#define gpuFree(p)              { int gid; cudaGetDevice(&gid); \
                                  fprintf(stderr,"%-15.15s: call gpuFree(%p) <",__FUNCTION__,p); \
                                  checkCudaErrors(cudaFree(p)); fprintf(stderr,">\n"); }


#undef  gpuHostMalloc
#undef  gpuHostFree

#define gpuHostMalloc(p,s,f)    { int gid; cudaGetDevice(&gid); \
                                  fprintf(stderr,"%-15.15s: GPU%d call gpuHostMalloc(%p,%d,%d) <",__FUNCTION__,gid,p,s,f); \
                                  checkCudaErrors(__cudaHostMalloc((void **) (p),s,f)); fprintf(stderr,"*p=%p>\n",*p); }
#define gpuHostFree(p,f)        { int gid; cudaGetDevice(&gid); \
                                  fprintf(stderr,"%-15.15s: GPU%d call gpuHostFree(%p,%d) <",__FUNCTION__,gid,p,f); \
                                  checkCudaErrors(__cudaHostFree(p,f)); fprintf(stderr,">\n"); }

#endif

#endif

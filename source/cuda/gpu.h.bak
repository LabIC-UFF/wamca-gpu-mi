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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/gpu_helper.h>
#include <cuda/gpu_string.h>

#define KERNEL      __global__
#define DEVICE      __device__
#define HOST        __host__

/*!
 * Sahred memory aligment
 */
#define GPU_SHARED_ALIGN         4
/*!
 * Global memory aligment
 */
#define GPU_GLOBAL_ALIGN         128


#ifndef __gpu_block_size
#define __gpu_block_size

/*!
 * Computes next block size multiple of \a block.
 *
 * @param   block   Block size
 * @param   size    Amount of memory
 * @return  Returns the next block size multiple of \a block.
 */
inline
size_t
gpuBlockSize(size_t size, size_t block) {
    return ((size / block) + ((size % block) != 0)) * block;
}

#define gpuSharedBlockSize(sz)      gpuBlockSize(sz,GPU_SHARED_ALIGN)
#define gpuGlobalBlockSize(sz)      gpuBlockSize(sz,GPU_GLOBAL_ALIGN)

#endif


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


enum GPUCopyKind {
    GCPK_HOST2DEVICE = 1,
    GCPK_DEVICE2HOST = 2,
};

#ifndef GPU_CUDA_DISABLED

#define gpuMalloc(p,sz)             checkCudaErrors(cudaMalloc((void **) (p),sz))
#define gpuFree(p)                  checkCudaErrors(cudaFree(p))
#define gpuMallocHost(p,sz)         checkCudaErrors(cudaMallocHost((void **) (p),sz))
#define gpuFreeHost(p)              checkCudaErrors(cudaFreeHost(p))
#define gpuMemcpy(d,s,sz,k)         checkCudaErrors(cudaMemcpy(d,s,sz,k))
#define gpuMemset(d,v,sz)           checkCudaErrors(cudaMemset(d,v,sz))
#define gpuHostToDevice(d,s,sz)     checkCudaErrors(cudaMemcpy(d,s,sz,cudaMemcpyHostToDevice))
#define gpuDeviceToHost(d,s,sz)     checkCudaErrors(cudaMemcpy(d,s,sz,cudaMemcpyDeviceToHost))
#define gpuHostToSymbol(d,s,sz)     checkCudaErrors(cudaMemcpyToSymbol(d,s,sz))
#define gpuMemInfo(f,t)             checkCudaErrors(cudaMemGetInfo(f,t))
#define gpuDevProperties(p,d)       checkCudaErrors(cudaGetDeviceProperties(p,d))
#define gpuDevSynchronize()         checkCudaErrors(cudaDeviceSynchronize())

#else

#define gpuMalloc(p,sz)             *(p) = NULL
#define gpuFree(p)                   (p) = NULL
#define gpuMallocHost(p,sz)         *(p) = NULL
#define gpuFreeHost(p)               (p) = NULL
#define gpuMemcpy(d,s,sz,k)
#define gpuMemset(d,v,sz)
#define gpuHostToDevice(d,s,sz)
#define gpuDeviceToHost(d,s,sz)
#define gpuHostToSymbol(d,s,sz)
#define gpuMemInfo(f,t)             *(f) = *(t) = 0
#define gpuDevProperties(p,d)
#define gpuDevSynchronize()

#endif  // GPU_CUDA_DISABLED

#endif

/**
 * @file   mkutil.cu
 *
 * @brief  CUDA utility code.
 *
 * @author Eyder Rios
 * @date   2014-06-01
 */

#include <iostream>
#include <stdint.h>
#include <stdarg.h>
#include "gpu.h"
#include "mlads.h"
#include "log.h"


#include "mlkernel.h" // IGOR add
//#include "mlgputask.h"



// ################################################################################# //
// ##                                                                             ## //
// ##                               KERNEL FUNCTIONS                              ## //
// ##                                                                             ## //
// ################################################################################# //


// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //


// ################################################################################ //
// ##                                                                            ## //
// ##                                  DATA TYPES                                ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                                    KERNEL                                  ## //
// ##                                                                            ## //
// ################################################################################ //


#define by      blockIdx.y
#define tx      threadIdx.x

//kernelShowData<<<1,1,0,stream>>>(gpuTask.gpuId,id,adsBuffer,width,max);

/*!
 * Show GPU data on screen.
 */
__global__
void
kernelShowData(int gid, int kid, MLADSData *ads, uint width, uint max)
{
    uint        i,j;
    uint       *data;

    if(max == 0)
        max = ads->s.solElems;
    else
    if(ads->s.solElems < max)
        max = ads->s.solElems;

    printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
    printf("Kernel %u:\n",kid);
    printf("\tGPU id\t: %u\n",gid);
    printf("\tglobal\t: %d\n",GPU_GLOBAL_ALIGN);
    printf("\tshared\t: %d\n\n",GPU_SHARED_ALIGN);

    printf("Kernel Data:\n");

    printf("\tsolSize\t\t: %6u elems\n",ads->s.solElems);
    printf("\tadsRowElems\t: %6u elems\t%6lu bytes\n",ads->s.rowElems,ads->s.rowElems * sizeof(uint));
    printf("\tadsData\t\t: %6lu elems\t%6u bytes\t%p\n",ads->s.size / sizeof(uint),ads->s.size,ads);
    printf("\tadsCoords\t: %6u elems\t%6lu bytes\t%p\n",ads->s.rowElems,ads->s.rowElems * sizeof(uint),ADS_COORD_PTR(ads));
    printf("\tadsSolution\t: %6u elems\t%6lu bytes\t%p\n",ads->s.rowElems,ads->s.rowElems * sizeof(uint),ADS_SOLUTION_PTR(ads,ads->s.rowElems));
    printf("\tadsTime\t\t: %6u elems\t%6lu bytes\t%p\n",ads->s.rowElems,ads->s.rowElems * sizeof(uint),ADS_TIME_PTR(ads,ads->s.rowElems));
    printf("\tadsCost\t\t: %6u elems\t%6lu bytes\t%p\n",ads->s.rowElems * ads->s.solElems,ads->s.rowElems * sizeof(uint) * ads->s.solElems,ADS_COST_PTR(ads,ads->s.rowElems));

    if(ads) {
        printf("\nSolution:\n");
        printf("\tcost\t: %u\n",ads->s.solCost);

        printf("\tclients\t: ");
        data = ADS_SOLUTION_PTR(ads,ads->s.rowElems);
        for(i=0;i < ads->s.solElems;i++)
            printf("%3hu",GPU_HI_USHORT(data[i]));
        printf("\n\tweights\t: ");
        for(i=0;i < ads->s.solElems;i++)
            printf("%3hu",GPU_LO_USHORT(data[i]));

        printf("\n\tcoord x\t: ");
        data = ADS_COORD_PTR(ads);
        for(i=0;i < ads->s.solElems;i++)
            printf("%3hu",GPU_HI_USHORT(data[i]));
        printf("\n\tcoord y\t: ");
        for(i=0;i < ads->s.solElems;i++)
            printf("%3hu",GPU_LO_USHORT(data[i]));
        printf("\n");

        printf("\nEval data (T):\n");
        data = ADS_TIME_PTR(ads,ads->s.rowElems);
        printf("%-*s:",width,"row");
        for(i=0;i < max;i++)
            printf("%*u",width,data[i]);
        printf("\n\n");

        for(i=0;i < max;i++) {
            for(j=0;j < max;j++) {
                if(j == 0)
                    printf("%*u:",width,i);
                if(i < j)
                    printf("%*u",width,data[j] - data[i]);
                else
                    printf("%*u",width,data[i] - data[j]);
            }
            printf("\n");
        }

        printf("\nEval data (C):\n");
        data = ADS_COST_PTR(ads,ads->s.rowElems);
        for(i=0;i < max;i++) {
            uint *row = data + i*ads->s.rowElems;

            for(j=0;j < max;j++) {
                if(j == 0)
                    printf("%*u:",width,i);
                printf("%*u",width,row[j]);
            }
            printf("\n");
        }
    }
    printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"

/*!
 * Empty kernel.
 */
__global__
void
kernelShowDist(const MLADSData *ads, int *buffer)
{
    __shared__
    int      sm_coordx[1024],
             sm_coordy[1024];
    int      i,j,d,size,dx,dy;
    uint    *coords,*sol;
    float    df,dr;
    int     *p;

    size   = ads->s.solElems;
    dr     = ads->s.round * 0.5F;

    coords = ADS_COORD_PTR(ads);
    sol    = ADS_SOLUTION_PTR(ads,size);

    for(i=0;i < size;i++) {
        sm_coordx[i] = GPU_HI_USHORT(coords[i]);
        sm_coordy[i] = GPU_LO_USHORT(coords[i]);

//        d = GPU_HI_USHORT(sol[i]);
//        kprintf("%d\t%d\t(%d,%d)\n",i,d,sm_coordx[i],sm_coordy[i]);
    }

    i = 0;
    j = 4;

    printf("\n\n");
    printf("round=%0.2f\n",dr);

    printf("P%d=(%d,%d)\n",i,sm_coordx[i],sm_coordy[i]);
    printf("P%d=(%d,%d)\n\n",j,sm_coordx[j],sm_coordy[j]);

    dx = sm_coordx[i] - sm_coordx[j];
    dy = sm_coordy[i] - sm_coordy[j];
    printf("dx=%d\t\tdy=%d\n",dx,dy);

    dx = dx * dx;
    dy = dy * dy;
    printf("dx2=%d\tdy2=%d\n",dx,dy);

    dx = dx + dy;
    printf("dx2 + dy2=%d\n",dx);

    df  = sqrtf(dx);
    d   = int(df + dr);
    printf("sqrtf(%d) = %0.4f\t%d\n",dx,df,d);

    printf("\n\n");

/*
    p = buffer;

    *p++ = size = 20;
    for(i=0;i < size;i++) {
        for(j=0;j < i;j++) {
            d = GPU_DIST_COORD(i,j);

            dx = sm_coordx[i] - sm_coordx[j];
            dy = sm_coordy[i] - sm_coordy[j];
            d  = sqrtf(dx*dx + dy*dy) + 0.5;

            df = sqrtf((float) ( (sm_coordx[i] - sm_coordx[j])*(sm_coordx[i] - sm_coordx[j]) ) +
                               ( (sm_coordy[i] - sm_coordy[j])*(sm_coordy[i] - sm_coordy[j]) ));
            *p++ = d;

            printf("%d\t%d\t(%d,%d)\t(%d,%d)\t%d\t%f\n",
                            i,j,
                            sm_coordx[i],sm_coordy[i],
                            sm_coordx[j],sm_coordy[j],
                            d,df);
        }
    }
*/
}

#pragma GCC diagnostic pop

/*!
 * Empty kernel.
 */
__global__
void
kernelEmpty(uint kid)
{
    kprintf("Kernel #%u\n",kid);
}

__global__
void
kernelSleep(uint kid, ullong count)
{
    ullong  start,
            offset;

    kprintf("Kernel #%u will sleep for %lluK clock cycles\n",kid,count);

    count *= 1000;

    start = clock64();
    offset = 0;
    while (offset < count) {
        offset = clock64() - start;
    }

    kprintf("Kernel #%u finished\n",kid);
}

__global__
void
kernelTest(uint kid, MLADSData *ads)
{
    extern  __shared__
    uint    buffer[];

    uint   *data,
           *p;

    if(tx == 0)
        kprintf("Kernel #%u: block %d\n",kid,by);

    data = ADS_TIME_PTR(ads,ads->s.rowElems) + by*ads->s.rowElems;
    p    = buffer + by * ads->s.rowElems;

    p[tx] = data[tx];
}

__global__
void
kernelWeight(const MLADSData *ads)
{
    __shared__ int sm_coordx[1024], sm_coordy[1024];
    float    sm_round;
    uint    *gm_data;

    sm_round = ads->s.round * 0.5F;

    gm_data = ADS_COORD_PTR(ads);
    sm_coordx[threadIdx.x] = GPU_HI_USHORT(gm_data[threadIdx.x]);
    sm_coordy[threadIdx.x] = GPU_LO_USHORT(gm_data[threadIdx.x]);

    __syncthreads();

//    if(blockIdx.x == 0)
//        kprintf("%d (%d,%d)\n",threadIdx.x,sm_coordx[threadIdx.x],sm_coordy[threadIdx.x]);

    gm_data = ADS_COST_PTR(ads,ads->s.rowElems) +
              blockIdx.x*ads->s.solElems;

    gm_data[threadIdx.x] = GPU_DIST_COORD(blockIdx.x,threadIdx.x);

    k4printf("D[%d,%d] = %d\t(%d,%d)-(%d,%d)\n",
                    blockIdx.x,threadIdx.x,gm_data[threadIdx.x],
                    sm_coordx[blockIdx.x],sm_coordy[blockIdx.x],
                    sm_coordx[threadIdx.x],sm_coordy[threadIdx.x]);
}

__global__
void
kernelChecksum(const MLADSData *ads, uint max, bool show)
{
    ulong  sum;
    byte  *data;

    k4printf("grid(%u,%u,%u)\n",gridDim.x,gridDim.y,gridDim.z);
    k4printf("block(%u,%u,%u)\n",blockDim.x,blockDim.y,blockDim.z);

    if(max == 0)
        max = ads->s.size;

    data = (byte *) ads;
    sum  = 0;
    for(uint i=0;i < max;i++)
        sum += data[i];

    kprintf("ADS_CS: %lu (GPU)\n",sum);
    if(show) {
        for(uint i=0;i < max;i++)
            kprintf(" %02x",data[i]);
        kprintf("\n");
    }
}

#undef by
#undef tx


// ################################################################################# //
// ##                                                                             ## //
// ##                               KERNEL LAUNCHERS                              ## //
// ##                                                                             ## //
// ################################################################################# //

void
MLKernel::launchEmptyKernel()
{
    gpuEventRecord(evtStart,stream);

    kernelEmpty<<<1,1,0,stream>>>(
                    id
    );

    gpuEventRecord(evtStop,stream);
    gpuEventSynchronize(evtStop);
}

void
MLKernel::launchSleepKernel(ullong time)
{
    gpuEventRecord(evtStart,stream);

    kernelSleep<<<1,1,0,stream>>>(
                    id,
                    time
    );

    gpuEventRecord(evtStop,stream);
    gpuEventSynchronize(evtStop);
}

void
MLKernel::launchTestKernel()
{
    grid.x = 1;
    grid.y = 5;
    grid.z = 1;

    block.x = adsRowElems;
    block.y = 1;
    block.z = 1;

    shared = block.x * grid.y * sizeof(uint);

    gpuEventRecord(evtStart,stream);

    kernelTest<<<grid,block,shared,stream>>>(
                    id,
                    adsData
    );

    gpuEventRecord(evtStop,stream);
    gpuEventSynchronize(evtStop);
}

void
MLKernel::launchShowDataKernel(uint width, uint max)
{
    gpuEventRecord(evtStart,stream);

    int gpuId = 0; // TODO: fix

    kernelShowData<<<1,1,0,stream>>>(
                    gpuId,
                    id,
                    adsData,
                    width,
                    max
    );

    gpuEventRecord(evtStop,stream);
    gpuEventSynchronize(evtStop);
}

void
MLKernel::launchWeightKernel()
{
    int       *weights,
              *row,
               count,
               i,j;

    gpuEventRecord(evtStart,stream);

    kernelWeight<<<solSize,solSize,0,stream>>>(adsData);

    gpuEventRecord(evtStop,stream);
    gpuEventSynchronize(evtStop);

    count   = solSize * solSize * sizeof(*weights);
    weights = new int[solSize * solSize];

    gpuMemcpy(weights,ADS_COST_PTR(adsData,adsRowElems),count,cudaMemcpyDeviceToHost);

#if 0
    printf("CPU\n");
    for(i=0;i < solSize;i++) {
        for(j=0;j < solSize;j++)
            printf("%4d",problem.clients[i].weight[j]);
        printf("\n");
    }

    printf("\nGPU\n");
    for(i=0;i < solSize;i++) {
        row = weights + i*solSize;
        for(j=0;j < solSize;j++)
            printf("%4d",row[j]);
        printf("\n");
    }
    printf("\n");
#endif

    count = 0;
    for(i=0;i < solSize;i++) {
        row = weights + i*solSize;
        for(j=0;j < solSize;j++) {
            if(problem.clients[i].weight[j] != row[j]) {
                count++;
                printf("D[%d,%d]\t: CPU=%d\tGPU=%d\n",
                                i,j,
                                problem.clients[i].weight[j],
                                row[j]);
            }
        }
    }

    printf("%d\t%s\n",count,problem.name);

    delete[] weights;
}

void
MLKernel::launchChecksumKernel(uint max, bool show)
{
    kernelChecksum<<<1,1,0,stream>>>(
                    adsData,
                    max,
                    show
    );
}

void
MLKernel::launchShowDistKernel()
{
    int  *h_buffer,*d_buffer,*p;
    int   i,j,n,siz;

    gpuEventRecord(evtStart,stream);

    siz = solSize*solSize*sizeof(int);
    h_buffer = new int[siz];
    gpuMalloc(&d_buffer,siz);

    l4printf("Launching %s\n",__FUNCTION__);
    kernelShowDist<<<1,1,0,stream>>>(adsData,d_buffer);

    gpuEventRecord(evtStop,stream);
    gpuEventSynchronize(evtStop);

    gpuMemcpy(h_buffer,d_buffer,siz,cudaMemcpyDeviceToHost);

    p = h_buffer;
    n = *p++;

//    lprintf("%d 0\n",n);
//    for(i=0;i < 20;i++) {
//        lprintf("%d",i);
//        for(j=0;j < i;j++)
//            lprintf(" %d",*p++);
//        printf("\n");
//    }

    delete[] h_buffer;
    gpuFree(d_buffer);
}


#include <gtest/gtest.h>
//
#include <stdio.h>
#include <sys/time.h>

#include "TestKernel.cuh"

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<class T>
//__global__ void test_mykernel(const T& func, int x)
__global__ void test_mykernel(T func, int x)
{
	int y = 0;
	//y += func(x);
	printf("y=%d\n", y);
}

/*
./MyKernel.Test.cu(34): error: The closure type for a lambda ("lambda [](int)->int", defined at ./MyKernel.Test.cu:34) 
cannot be used in the template argument type of a __global__ function template instantiation, 
unless the lambda is defined within a __device__ or __global__ function, or the lambda is
 an 'extended lambda' and the flag --expt-extended-lambda is specified
          detected during instantiation of "test_mykernel" based on template argument <lambda [](int)->int> 
*/

// This is used to launch lambda in a 'clean' scope. Could also be a global function.
struct RunTestKernel
{
	static void run(int x)
	{
		std::cout << "will launch!" << std::endl;
		//
		// must use 'extended lambda' (with __device__) and allow flag '--expt-extended-lambda'
		test_mykernel<<<1,1>>>( [] __host__ __device__ (int k)->int { return 2*k; } , x);
		//
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		std::cout << "launched and finished!" << std::endl;
	}
};



//
//https://stackoverflow.com/questions/32567479/cuda-7-5-experimental-host-device-lambdas#36444843
//
template<class T> 
__global__ void kernel_myforeach( T func )
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    func( i );
}
//
template<bool onDevice, class T> void myforeach( size_t size, T func )
{
    if( onDevice )
    {
        size_t blocksize = 32;
        size_t gridsize = size/32;
        kernel_myforeach<<<gridsize,blocksize>>>( func );
    }
    else
        for( size_t i = 0; i < size; ++i )
            func( i );    
}
//
__global__ void printFirstElementOnDevice( double* v ) { printf( "dVector[0] = %f\n", v[0] ); }
//
template<bool onDevice> void assignScalar( size_t size, double* v, double a )
{
    //auto assign = [=]  __host__ __device__ ( size_t i ) { v[i] = a; };
	//myforeach<onDevice>( size, assign );
	myforeach<onDevice>( size, [=]  __host__ __device__ ( size_t i ) { v[i] = a; } );
}

//https://stackoverflow.com/questions/32567479/cuda-7-5-experimental-host-device-lambdas#36444843
//
TEST(TestMyKernel, WamcaExperiment_Lambda_CPP14_example)
{
	constexpr size_t SIZE = 32;

    double* hVector = new double[SIZE];
    double* dVector;
    cudaMalloc( &dVector, SIZE*sizeof(double) );

    for( size_t i = 0; i < SIZE; ++i )
        hVector[i] = 0;
    
    cudaMemcpy( dVector, hVector, SIZE*sizeof(double), cudaMemcpyHostToDevice );

	assignScalar<false>( SIZE, hVector, 3.0 );
	EXPECT_EQ(hVector[0], 3);
    //std::cout << "hVector[0] = " << hVector[0] << std::endl;

    assignScalar<true>( SIZE, dVector, 4.0 );
    printFirstElementOnDevice<<<1,1>>>( dVector );
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
	EXPECT_EQ(error, cudaSuccess);
}


TEST(TestMyKernel, WamcaExperiment_Lambda_CPP14_test)
{
	//
	test_mykernel2<<<1,1>>>(50);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );


	//
	int x = 50;
	// cannot run from this scope... cuda requires a 'clean' scope, to use 'extended lambdas'
	RunTestKernel::run(x);
	cudaError_t code = cudaDeviceSynchronize();
	EXPECT_EQ(code, cudaSuccess);
	//
	std::cout << "last kernel passed" << std::endl;
}

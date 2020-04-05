
#include "TestKernel.cuh"

#include<stdio.h>

__global__ void test_mykernel2(int x)
{
	printf("x=%d\n", x);
}

/*
// will cause error
__global__ void test_mykernel1(int& x)
{
	x++;
	printf("x=%d\n", x);
}
*/


template<class T>
__global__ void test_mykernel(const T& func, int x)
{
	int y = func(x);
	printf("y=%d\n", y);
}

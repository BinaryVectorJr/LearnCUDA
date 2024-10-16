#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 10000000
#define MAX_ERR 1e-6

/// <summary>
/// CUDA Hello World
/// </summary>
/// <returns></returns>
//__global__ void cuda_hello()
//{
//	printf("Hello World from GPU");
//}

/// <summary>
/// CPU Add Vectors
/// </summary>
/// <returns></returns>
//void vector_add(float* out, float* a, float* b, int n)
//{
//	for (int i = 0; i < n; i++)
//	{
//		out[i] = a[i] + b[i];
//	}
//}

/// <summary>
/// CUDA Vector Addition (vector add on single thread)
/// </summary>
/// <returns></returns>
//__global__ void vector_add_cuda(float* out, float* a, float* b, int n)
//{
//	for (int i = 0; i < n; i++)
//	{
//		out[i] = a[i] + b[i];
//	}
//}

/// <summary>
/// CUDA Vector Addition - multiple threads
/// CUDA organizes threads into a group called "Thread Block"
/// Kernal can launch multiple thread blocks, all organized into a "grid" structure
/// </summary>
/// <returns></returns>
//__global__ void vector_add_cuda_multi(float* out, float* a, float* b, int n)
//{
//	// Getting the index of the thread within the block
//	// Usually indexes are from 0 to 255
//	int index = threadIdx.x;
//	// Getting the size of the thread block i.e. number of threads in TB
//	// Usually stride increases by a value of 256
//	int stride = blockDim.x;
//
//	for (int i = index; i < n; i += stride)
//	{
//		out[i] = a[i] + b[i];
//	}
//}

/// <summary>
/// CUDA Vector Addition Multi-Threaded Multi-Thread Blocks
/// </summary>
/// <returns></returns>
__global__ void vector_add_cuda_grid(float *out, float *a, float *b, int n)
{
	// Unique index of each Thread Block is needed
	// For e.g. 256 threads per thread block, we need (N/256) thread blocks to have N threads
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	// Handling arbitary vector size
	if (threadID < n)
	{
		out[threadID] = a[threadID] + b[threadID];
	}

}


int main()
{
	/// CUDA Hello World
		//cuda_hello <<<1, 1 >>> ();
		//return 0;
	///-----------------------------------
	
	/// CPU Vector Add
		//float* a, * b, * out;

		//// Allocate memory
		//a = (float*)malloc(sizeof(float) * N);
		//b = (float*)malloc(sizeof(float) * N);
		//out = (float*)malloc(sizeof(float) * N);

		////Initialize Array
		//for (int i = 0; i < N; i++)
		//{
		//	a[i] = 1.0f;
		//	b[i] = 2.0f;
		//}

		//// Call Vector Add main function
		//vector_add(out, a, b, N);
	///-------------------------------------

	/// CUDA Vector Add (Same for CUDA Vector Add Multi)
	float* a, * b, * out;
	float* dev_a, *dev_b, *dev_out;

	// Allotting the host memory for array a, b and the output array
	a = (float*)malloc(sizeof(float) * N);
	b = (float*)malloc(sizeof(float) * N);
	out = (float*)malloc(sizeof(float) * N);

	/// MISSED - Initializing Host Arrays
	for (int i = 0; i < N; i++)
	{
		a[i] = 1.0f;
		b[i] = 2.0f;
	}

	/// Allotting the device memory for array a, b, and output array
	cudaMalloc((void**)&dev_a, sizeof(float) * N);
	cudaMalloc((void**)&dev_b, sizeof(float) * N);
	cudaMalloc((void**)&dev_out, sizeof(float) * N);

	/// Transfer data from host memory to device memory
	cudaMemcpy(dev_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

	/// Execute single thread kernel
	// vector_add_cuda << <1, 1 >> > (dev_out, dev_a, dev_b, N);

	/// Execute multi-thread kernel
	/// 1 = M = number of thread blocks, 256 = T = no of threads per block
	//vector_add_cuda_multi << <1, 256 >> > (dev_out, dev_a, dev_b, N);

	/// Execute multi-thread block kernel
	/// This is PARALLELIZING a piece of code - to work on multiple TBs with "T" number of threads
	int block_size = 256; //T
	int grid_size = ((N + block_size) / block_size);
	vector_add_cuda_grid << <grid_size, block_size >> > (dev_out, dev_a, dev_b, N);

	/// Transfer data back from device memory to host memory
	cudaMemcpy(out, dev_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

	/// MISSED - Verification
	for (int i = 0; i < N; i++)
	{
		assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
	}
	printf("out[0] = %f\n", out[0]);
	printf("PASSED\n");

	/// Cleaning up the pointers (deallocating) after kernel execution
	/// Need to do for both host and device
	
	/// Deallocate device memory first (together) then host memory (together)
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_out);

	/// Deallocate host memory
	free(a);
	free(b);
	free(out); // MISSED
}
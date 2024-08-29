#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define N 10000000

// CUDA Hello World
__global__ void cuda_hello()
{
	printf("Hello World from GPU");
}

// CPU Add Vectors
void vector_add(float* out, float* a, float* b, int n)
{
	for (int i = 0; i < n; i++)
	{
		out[i] = a[i] + b[i];
	}
}

int main()
{
	///CUDA Hello World
	//cuda_hello <<<1, 1 >>> ();
	//return 0;

	float* a, * b, * out;

	// Allocate memory
	a = (float*)malloc(sizeof(float) * N);
	b = (float*)malloc(sizeof(float) * N);
	out = (float*)malloc(sizeof(float) * N);

	//Initialize Array
	for (int i = 0; i < N; i++)
	{
		a[i] = 1.0f;
		b[i] = 2.0f;
	}

	// Call Vector Add main function
	vector_add(out, a, b, N);
}
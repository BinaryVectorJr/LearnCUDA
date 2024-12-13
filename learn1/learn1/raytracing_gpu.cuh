#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// Definitions to bypass red squiggly lines when doing kernel calls
// https://forums.developer.nvidia.com/t/intellisense-error-in-brand-new-cuda-project-in-vs2019/111921/7
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< grids, threads >>>
#define KERNEL_ARGS3(grid, block, sh_mem)         <<< grids, threads, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grids, threads, sh_mem, stream >>>
#define __syncthreads()
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#define __syncthreads()
#endif

// Custom definitions (infinty, rand, number of spheres)
#define DIM 1024 // --> this goes max till 2048
#define NUM_SPHERES 20 // --> tested till 50
#define INF 2e10f
#define rnd(x)(x*rand()/RAND_MAX)

// globals needed by the update routine
struct DataBlock {
	unsigned char* dev_bitmap;
};

// Defining custom structure for sphere
struct Sphere
{
	// Color values
	float r, b, g;

	// Radius of sphere
	float radius;

	// Location in 3D space
	float x, y, z;
	/// <summary>
	/// Compute whether the ray intersect the sphere, and if so then compute the distance from the pixel to the camera where it hits
	/// </summary>
	/// <param name="ox">X location of pixel</param>
	/// <param name="oy">Y location of pixel</param>
	/// <param name="n"></param>
	/// <returns></returns>
	__device__ float hit(float ox, float oy, float* n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if (dx * dx + dy * dy < radius * radius)
		{
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};

// Variable NOT in constant memory
// Sphere* uSphere;

// Varaible in constant memory
__constant__ Sphere uSphere[NUM_SPHERES];

// Main Kernel
__global__ void raytracing_gpu(unsigned char* ptr);

// Main Program
void GPU_RayTracing();



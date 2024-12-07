// GPU Implementation

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

// Link external libraries: https://www.programmersought.com/article/5070844086/

#define DIM 1000

// Definitions to bypass red squiggly lines when doing kernel calls
// https://forums.developer.nvidia.com/t/intellisense-error-in-brand-new-cuda-project-in-vs2019/111921/7
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem)         <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif


// CUDA Kernel - Called by the Host and run on the Device
__global__ void GPU_Kernel(unsigned char *_ptr);

// Device specific function (and is not accessible from the CPU)
__device__ int GPU_JuliaCalc(int _x, int _y);

// The "main" function
void GPU_Generate_JuliaSet();

// To store the device bitmap data, should we need it
struct DataBlock {
	unsigned char *dev_bitmap;
};

/// <summary>
/// Generic structure to store complex numbers
/// </summary>
struct GPU_cuComplex
{
	float r;	// Real Component of complex number
	float i;	// Imaginary Component of complex number

	// Complex number constructor on the GPU (this also needs a __device__ which is not mentioned in the book)
	__device__ GPU_cuComplex(float _a, float _b) : r(_a), i(_b) {}

	// Calculate the magnitude of a complex number on the GPU
	__device__ float magnitude2(void) 
	{ 
		return r * r + i * i; 
	}

	// Defining the '*' operation for complex numbers (MULTIPLICATION) on the GPU
	__device__ GPU_cuComplex operator*(const GPU_cuComplex& a)
	{
		return GPU_cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}

	// Defining the '+' operation for complex numbers (ADDITION) on the GPU
	__device__ GPU_cuComplex operator+(const GPU_cuComplex& a)
	{
		return GPU_cuComplex(r + a.r, i + a.i);
	}
};
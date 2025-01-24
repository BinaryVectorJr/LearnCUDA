#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define DIM 1000
#define SPEED 5

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

// ==============================================

__global__ void heattransfer_gpu()
{

}

/// STEP 1
// Given the grid of input (starter) temps, copy temp. of cells into the input grid of the main kernel
__global__ void copy_const_kernel(float* iptr, const float* cptr)
{	

	// Do mapping from threadIdx or BlockIdx to x and y-coordinate of the window
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// Linear offset into our buffers
	int offset = x + y * blockDim.x * gridDim.x;

	// Copy temp. from cptr - constatnt to iptr - input
	// Copy is performed only if cell has non-zero value in const. grid
	// This is done to preserve temps for any non-zero temps in the input for each step
	if (cptr[offset] != 0)
	{
		iptr[offset] = cptr[offset];
	}

}

/// STEP 2
// Each thread takes up responsibility for 1 cell of our grid
// Thread reads temps of that cell, temps of surrounding cells, computes updates, and updates the cell's temp. with a new value
__global__ void blend_kernel(float* outSrc, const float* inSrc)
{
	// Do mapping from threadIdx or BlockIdx to x and y-coordinate of the window
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// Linear offset into our buffers
	int offset = x + y * blockDim.x * gridDim.x;

	// Calculating the positions of the neighbors, based on offset value since we use offset value as index
	int left = offset - 1;
	int right = offset + 1;
	// Now here, we need to adjust for border indexes so that cells on the edges do not wrap around
	// Accounting for first value of x (first row of grid)
	if (x == 0)
		left++;
	// Accounting for last value of x (last row of grid)
	if (x == DIM - 1)
		right++;

	int top = offset - DIM;
	int bottom = offset + DIM;
	// Accounting for first value of y (first column of grid)
	if (y == 0)
		top += DIM;
	// Accounting for last value of y (last column of grid)
	if (y == DIM - 1)
		bottom -= DIM;

	outSrc[offset] = inSrc[offset] + SPEED * (
		inSrc[top] + inSrc[bottom] + inSrc[left] + inSrc[right]
		- inSrc[offset] * 4);
}
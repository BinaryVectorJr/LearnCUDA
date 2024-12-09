#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>	// For syncthreads method

// Definitions to bypass red squiggly lines when doing kernel calls
// https://forums.developer.nvidia.com/t/intellisense-error-in-brand-new-cuda-project-in-vs2019/111921/7
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< blocksPerGrid, threadsPerBlock >>>
#define KERNEL_ARGS3(grid, block, sh_mem)         <<< blocksPerGrid, threadsPerBlock, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< blocksPerGrid, threadsPerBlock, sh_mem, stream >>>
//#define __syncthreads()
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
//#define __syncthreads()
#endif

// Define an inline function to return the lesser of two numbers
#define imin(a,b) (a<b?a:b)

// Calculating sum of squares for dot product
#define sumSquares(x) (x*(x+1)*(2*x+1)/6)

// GPU Kernel
__global__ void dotproduct_gpu(float* a, float* b, float* c);

// Main program
void GPU_CalcDotProduct();


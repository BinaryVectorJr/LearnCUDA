#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cpu_bitmap.h>
#include <device_functions.h>	// For syncthreads method

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

// ============= END OF BOILERPLATE =========================

#define DIM 1024
# define PI 3.1415926535897932f

// Main bitmap in shared memory program
void GPU_SharedMemoryBitmap();

// GPU Kernel
__global__ void sharedmembitmap_gpu(unsigned char *_ptr);


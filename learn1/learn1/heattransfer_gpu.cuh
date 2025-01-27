// Skip Ch7 since CUDA 12.6 does not support "texture" type, rather cudaTextureObject_t. Learn about it separately

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "./cpu_anim.h"
//#include "./book.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

// Definitions to bypass red squiggly lines when doing kernel calls
// https://forums.developer.nvidia.com/t/intellisense-error-in-brand-new-cuda-project-in-vs2019/111921/7
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(blocks, threads)                 <<< blocks, threads >>>
#define KERNEL_ARGS3(blocks, threads, sh_mem)         <<< blocks, threads, sh_mem >>>
#define KERNEL_ARGS4(blocks, threads, sh_mem, stream) <<< blocks, threads, sh_mem, stream >>>
#define __syncthreads()
#else
#define KERNEL_ARGS2(blocks, threads)
#define KERNEL_ARGS3(blocks, threads, sh_mem)
#define KERNEL_ARGS4(blocks, threads, sh_mem, stream)
#define __syncthreads()
#endif

// ==============================================

// Structure to store data for the update routine to work on which encapsulates the entire state of animation without using "ticks" anywhere
struct DataBlock
{
	unsigned char* output_bitmap;
	float* device_inSrc;
	float* device_outSrc;
	float* device_constSrc;
	CPUAnimBitmap* bitmap;
	cudaEvent_t start, stop;
	float totalTime;
	float frames;
};

// Texture variables on the GPU


int GPU_HeatTransferMain(void);
void heattransfer_anim_gpu(DataBlock* d, int ticks);
void heattransfer_anim_exit(DataBlock* d);
template< typename T >
void swap(T& a, T& b) 
{
	T t = a;
	a = b;
	b = t;
}

__global__ void copy_const_kernel(float* iptr, const float* cptr);
__global__ void blend_kernel(float* outSrc, const float* inSrc);

__device__ unsigned char value(float n1, float n2, int hue);
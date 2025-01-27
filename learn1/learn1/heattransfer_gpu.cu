// Skip Ch7 since CUDA 12.6 does not support "texture" type, rather cudaTextureObject_t. Learn about it separately

#include "./heattransfer_gpu.cuh"

__device__ unsigned char value(float n1, float n2, int hue) {
	if (hue > 360)      hue -= 360;
	else if (hue < 0)   hue += 360;

	if (hue < 60)
		return (unsigned char)(255 * (n1 + (n2 - n1) * hue / 60));
	if (hue < 180)
		return (unsigned char)(255 * n2);
	if (hue < 240)
		return (unsigned char)(255 * (n1 + (n2 - n1) * (240 - hue) / 60));
	return (unsigned char)(255 * n1);
}

__global__ void float_to_color(unsigned char* optr,
	const float* outSrc) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float l = outSrc[offset];
	float s = 1;
	int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
	float m1, m2;

	if (l <= 0.5f)
		m2 = l * (1 + s);
	else
		m2 = l + s - l * s;
	m1 = 2 * l - m2;

	optr[offset * 4 + 0] = value(m1, m2, h + 120);
	optr[offset * 4 + 1] = value(m1, m2, h);
	optr[offset * 4 + 2] = value(m1, m2, h - 120);
	optr[offset * 4 + 3] = 255;
}

__global__ void float_to_color(uchar4* optr,
	const float* outSrc) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float l = outSrc[offset];
	float s = 1;
	int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
	float m1, m2;

	if (l <= 0.5f)
		m2 = l * (1 + s);
	else
		m2 = l + s - l * s;
	m1 = 2 * l - m2;

	optr[offset].x = value(m1, m2, h + 120);
	optr[offset].y = value(m1, m2, h);
	optr[offset].z = value(m1, m2, h - 120);
	optr[offset].w = 255;
}

int GPU_HeatTransferMain(void)
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;
	// HANDLE_ERROR(cudaEventCreate(&data.start));
	// HANDLE_ERROR(cudaEventCreate(&data.stop));

	// HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, bitmap.image_size()));

	// Assuming that float == 4 chars in size for RGBA
	// HANDLE_ERROR(cudaMalloc((void**)&data.device_inSrc, bitmap.image_size()));
	// HANDLE_ERROR(cudaMalloc((void**)&data.device_outSrc, bitmap.image_size()));
	// HANDLE_ERROR(cudaMalloc((void**)&data.device_constSrc, bitmap.image_size()));

	float* temp = (float*)malloc(bitmap.image_size());
	for (int i = 0; i < DIM * DIM; i++)
	{
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
		{
			temp[i] = MAX_TEMP;
		}
	}

	temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM * 700 + 100] = MIN_TEMP;
	temp[DIM * 300 + 300] = MIN_TEMP;
	temp[DIM * 200 + 700] = MIN_TEMP;
	for (int y = 800l; y < 900; y++)
	{
		for (int x = 400; x < 500; x++)
		{
			temp[x + y * DIM] = MIN_TEMP;
		}
	}

	// HANDLE_ERROR(cudaMemcpy(data.device_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));

	for (int y = 800; y < DIM; y++)
	{
		for (int x = 0; x < 200; x++)
		{
			temp[x + y * DIM] = MAX_TEMP;
		}
	}

	// HANDLE_ERROR(cudaMemcpy(data.device_inSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));

	free(temp);

	bitmap.anim_and_exit((void (*)(void*, int))heattransfer_anim_gpu, (void (*)(void*))heattransfer_anim_exit);

	exit(0);
}

// Gets called every frame
// Data structure for CUDA and ticks of animation that have elapsed
void heattransfer_anim_gpu(DataBlock* d, int ticks)
{
	// Start recording events for event based timing
	// HANDLE_ERROR(cudaEventRecord(d->start, 0));
	dim3 blocks(DIM / 16, DIM / 16);
	// Using a block of 256 threads arranged 16x16
	dim3 threads(16, 16);
	CPUAnimBitmap* bitmap = d->bitmap;

	// Each for loop iteration computes a single time step in the three step algorithm (2 of the steps are defined below)
	// 90 value was experimentally found out to have reasonable tradeoff b/w downloading a bitmap image and computing too many steps, per frame
	for (int i = 0; i < 90; i++)
	{
		copy_const_kernel KERNEL_ARGS2(blocks, threads)(d->device_inSrc, d->device_constSrc);
		blend_kernel KERNEL_ARGS2(blocks, threads)(d->device_outSrc, d->device_inSrc);
		// Swap is important as it contains the output of the 90th time step - the for loop itself leaves the i/p & o/p swapped, so we have to do this
		swap(d->device_inSrc, d->device_outSrc);
		// After 90 time steps since previous frame, copy a bitmap frame of animation back to CPU
	}


	// Convert float value to color
	float_to_color KERNEL_ARGS2(blocks, threads) (d->output_bitmap, d->device_inSrc);

	// Copy resultant image back to CPU
	//HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(),d->output_bitmap,bitmap->image_size(),cudaMemcpyDeviceToHost));

	// HANDLE_ERROR(cudaEventRecord(d->stop, 0));
	// HANDLE_ERROR(cudaEventSynchronize(d->stop));

	float elapsedTime;
	// HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));

	//d->totalTime += elapsedTime;
	//++d->frames;
	//printf("Avg. time per frame: %3.1f ms\n", d->totalTime / d->frames);
}

void heattransfer_anim_exit(DataBlock* d)
{
	cudaFree(d->device_inSrc);
	cudaFree(d->device_outSrc);
	cudaFree(d->device_constSrc);

	// HANDLE_ERROR(cudaEventDestroy(d->start));
	// HANDLE_ERROR(cudaEventDestroy(d->stop));
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

	// Update the output dataset by adding the old temperature and the scaled differences of the neighbor's temps
	// Eqn 7.2: T_new = T_old + Sum_neighbors(k*T_neighbor - T_old) --> T_old and k are constants
	outSrc[offset] = inSrc[offset] + SPEED * (
		inSrc[top] + inSrc[bottom] + inSrc[left] + inSrc[right]
		- inSrc[offset] * 4);
}
#include "./sharedmembitmap_gpu.cuh"

// Kernel to compute pixel value for a single output location
__global__ void sharedmembitmap_gpu(unsigned char *_ptr)
{
	// Mapping values from threadIdx or blockIdx to each pixel's position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// Shared buffer to store computations
	__shared__ float sharedCache[16][16];

	// Calculation of value at a pixel position
	const float period = 128.0f;
	sharedCache[threadIdx.x][threadIdx.y] = 255 * (sinf(x * 2.0f * PI / period) + 1.0f) * (sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;

	// Syncing threads to eliminate race conditions - without this output is weird
	__syncthreads();

	_ptr[offset * 4 + 0] = 0;
	_ptr[offset * 4 + 1] = sharedCache[15-threadIdx.x][15-threadIdx.y];
	_ptr[offset * 4 + 2] = 0;
	_ptr[offset * 4 + 3] = 255;

}

// Main program to show bitmap with image values
void GPU_SharedMemoryBitmap()
{
	CPUBitmap memBitmap(DIM, DIM);
	unsigned char* device_bitmap;

	// Allocate memory on the device
	cudaMalloc((void**)&device_bitmap, memBitmap.image_size());

	// Creating grid for image
	dim3 grids(DIM / 16, DIM / 16);

	// Creating grid for threadblocks
	dim3 threads(16, 16);

	// Calling the GPU kernel
	sharedmembitmap_gpu KERNEL_ARGS2(grids, threads) (device_bitmap);

	// Copy bitmap value from GPU (Device) to CPU (Host)
	cudaMemcpy(memBitmap.get_ptr(), device_bitmap, memBitmap.image_size(), cudaMemcpyDeviceToHost);

	memBitmap.display_and_exit();

	cudaFree(device_bitmap);
}

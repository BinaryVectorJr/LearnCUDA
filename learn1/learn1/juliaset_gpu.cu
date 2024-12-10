// GPU Implementation

#include <cpu_bitmap.h>
#include "juliaset_gpu.cuh"

/// <summary>
/// Main proram to invoke calc. and viz. of a Julia set using the GPU
/// </summary>
void GPU_Generate_JuliaSet()
{
	DataBlock data;
	CPUBitmap uBitmap(DIM, DIM, &data);
	unsigned char *device_bitmap;

	// CUDA - allocating the memory on the device for the data
	if (cudaMalloc((void**)&device_bitmap, uBitmap.image_size()) != cudaSuccess)
	{
		std::cout << "cudaMalloc Error";
	}
	// Add ing the data to the custom structure
	data.dev_bitmap = device_bitmap;

	dim3 grid(DIM, DIM);
	// Fancy wrapper to eliminate squiggly line in the general kernel calls
	// GPU_Kernel << <grid, 1 >> > (device_bitmap);
	juliaset_gpu KERNEL_ARGS2(grid, 1) (device_bitmap);
	
	// TODO: I should change the above kernel call from 1 to 10 (or N) to increase performance
	// Try to depend less on launching blocks and try to have a balance b/w blocks and threads
	// Threads launch and execute serially if each block does not get a scheduler; threads always run simultaneously thus always having a scheduler.

	// CUDA - checking and printing the last error stored (if any)
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		std::cout << cudaGetErrorString(error);
	}

	// CUDA - copying the data from the memory set in the device (GPU) to the host (CPU)
	if (cudaMemcpy(uBitmap.get_ptr(), device_bitmap, uBitmap.image_size(), cudaMemcpyDeviceToHost))
	{
		std::cout << "cudaMemcpy Error";

	}

	// CUDA - Freeing the allocated memory
	cudaFree(device_bitmap);
	
	// Store values of bitmap to check which location has what values
	//std::vector<unsigned char> bitmapValues(uBitmap.pixels, uBitmap.pixels+uBitmap.image_size());

	// Displaying the Julia Set on a bitmap screen
	uBitmap.display_and_exit();
}

/// <summary>
/// CUDA Kernel to invoke the calculation of Julia Values and set the position and color
/// </summary>
/// <param name="_ptr"></param>
/// <returns></returns>
__global__ void juliaset_gpu(unsigned char *_ptr)
{
	// Mapping occurs from threadIdx or BlockIdx to position of a pixel

	int x = blockIdx.x;		// TODO: this would need to change based on whether we are using 1 or N threads in kernel call
	int y = blockIdx.y;		// TODO: this would also change based on whether we are using 1 or N threads in kernel call; 
	int offset = x + y * gridDim.x;

	// Calculating the value at that position (stride of 4)
	int GPU_JuliaValue = GPU_JuliaCalc(x, y);
	_ptr[offset * 4 + 0] = 255 * GPU_JuliaValue;
	_ptr[offset * 4 + 1] = 0;
	_ptr[offset * 4 + 2] = 0;
	_ptr[offset * 4 + 3] = 255;
}

/// <summary>
/// Julia Set calculation on the GPU
/// </summary>
/// <param name="_x">Width</param>
/// <param name="_y">Length</param>
/// <returns></returns>
__device__ int GPU_JuliaCalc(int _x, int _y)
{
	// To zoom in or zoom out
	const float scale = 1.5f;

	// Convert a give (x,y) image point to complex space point at (jx,jy)
	float jx = scale * (float)(DIM / 2 - _x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - _y) / (DIM / 2);

	// Arbitary complex constant based on iterative Julia Set equation
	GPU_cuComplex c(-0.8, 0.156);	//-0.8+0.156i
	GPU_cuComplex a(jx, jy);

	// Checking if a value belongs to the Julia Set
	// TODO: Reducing i by 50% increases performance slightly; check how to make throughput better (use NSight for metrics)
	int i = 0;
	for (i = 0; i < 200; i++)
	{
		a = a * a + c;
		if (a.magnitude2() > 1000)
		{
			return 0;
		}
	}
	return 1;
}
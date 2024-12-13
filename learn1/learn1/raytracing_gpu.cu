#include "./raytracing_gpu.cuh"
#include "./cpu_bitmap.h"

// Raytracing kernel
__global__ void raytracing_gpu(unsigned char* _ptr)
{
	// Mapping from threadIdx/blockIdx to position of pixels
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	// Calculating the stride by which memory spaces "jump"
	int offset = x + y * blockDim.x * gridDim.x;

	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);

	// Iterate through spheres to check for hits
	float r = 0, b = 0, g = 0;
	float maxz = -INF;
	for (int i = 0; i < NUM_SPHERES; i++)
	{
		float n;
		float t = uSphere[i].hit(ox, oy, &n);
		if (t > maxz)
		{
			float fScale = n;
			r = uSphere[i].r * fScale;
			b = uSphere[i].b * fScale;
			g = uSphere[i].g * fScale;
			maxz = t;
		}
	}

	// Store color in current output image
	_ptr[offset * 4 + 0] = (int)(r * 255);
	_ptr[offset * 4 + 1] = (int)(b * 255);
	_ptr[offset * 4 + 2] = (int)(g * 255);
	_ptr[offset * 4 + 3] = 255;
}

void GPU_RayTracing()
{
	DataBlock data;

	// Capturng the start and stop times to use CUDA event to check for timings
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	CPUBitmap bitmap(DIM, DIM, &data);
	unsigned char* device_bitmap;

	// Memory allocation on the GPU for output
	cudaMalloc((void**)&device_bitmap, bitmap.image_size());

	// Memory allocation for the sphere dataset (input data) on the GPU
	cudaMalloc((void**)&uSphere, sizeof(Sphere) * NUM_SPHERES);

	// Generate center coordinate, color, and radius randomly for all spheres
	// Allocating, initializing , and copy to memory in GPU, and then free temp memory
	// This is how you have to use custom structures
	Sphere* tempSphere = (Sphere*)malloc(sizeof(Sphere) * NUM_SPHERES);
	for (int i = 0; i < NUM_SPHERES; i++)
	{
		tempSphere[i].r = rnd(1.0f);
		tempSphere[i].b = rnd(1.0f);
		tempSphere[i].g = rnd(1.0f);
		tempSphere[i].x = rnd(1000.0f)-500;
		tempSphere[i].y = rnd(1000.0f)-500;
		tempSphere[i].z = rnd(1000.0f-500);
		tempSphere[i].radius = rnd(100.0f)+20;
	}

	// Line added to use access from constant memory
	cudaMemcpyToSymbol(uSphere, tempSphere, sizeof(Sphere) * NUM_SPHERES);


	// Copy buffer to GPU - not needed if we have __constant__
	// cudaMemcpy(uSphere, tempSphere, sizeof(Sphere) * NUM_SPHERES, cudaMemcpyHostToDevice);

	// Freeing the host memory
	free(tempSphere);

	// Input is now on GPU and space for output has been allocated, launch kernel
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16 , 16);
	raytracing_gpu KERNEL_ARGS2(grids, threads) (device_bitmap);

	// Copy bitmap back from GPU to CPU
	cudaMemcpy(bitmap.get_ptr(), device_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);


	// To get the end timestamp for the event of recording time
	cudaEventRecord(stop, 0);
	// This is needed since we cannot record the end time, until the GPU has finished all of its previous work - GPU may still be working on data when processing the "stop" function, thus the need for this function
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %3.1f ms\n", elapsedTime);

	// Destroy and cleanup events called
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Free allocated memory
	cudaFree(device_bitmap);

	bitmap.display_and_exit();
}

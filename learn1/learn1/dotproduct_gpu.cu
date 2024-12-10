#include "./dotproduct_gpu.cuh"

const int N = 33 * 1024;
const int threadsPerBlock = 256;

// For N data elements, we need N threads to compute dot product
// This allows us to set a value for the number of thread blocks we want to use
// And if we supply a smaller array, then the generic calculation can be used to get the smallest multiple we need
// This is a general trick and can be used in computations where we need to ensure that the most optimal value for the number of thread blocks is being used
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

// GPU kernel to calculate dot product of two arrays and put it in a resultant array
__global__ void dotproduct_gpu(float* a, float* b, float* c)
{
	// Creating buffer of shared memory to store the running sum of each thread
	__shared__ float sharedCache[threadsPerBlock];

	// Creating threadID (thread's index in a flattened array) using thread index, block index, and block dimension
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Creating index to use in shared cache
	int cacheIndex = threadIdx.x;

	float temp = 0;

	// Calculate for the entire length of the array
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		// Adding the stride to jump the id
		tid += blockDim.x * gridDim.x;
	}

	// Set values of the shared cache array
	sharedCache[cacheIndex] = temp;

	// Synchronize the threads in a block, to avoid race conditions (Intellisense does not detect syncthreads for some reason --> added to ifndef in .cuh)
	__syncthreads();

	// Reductions - reducing the dimension of inputs
	// threadsPerBlock should be a power of 2, so that the number of iterations to reduce to one value can be calculated for this specific code
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
		{
			sharedCache[cacheIndex] += sharedCache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	
	// When only one element is present in the running total array i.e. sharedCache[0]
	if (cacheIndex == 0)
	{
		// Write the value to the output array's blockId position
		c[blockIdx.x] = sharedCache[0];
	}
}

void GPU_CalcDotProduct()
{
	// Create the host memory space for arrays
	float* a, * b, * partial_c;

	// Create the device memory space for arrays
	float* device_a, * device_b, * device_partial_c;

	// CPU side memory allocation
	a = (float*)malloc(N * sizeof(float));
	b = (float*)malloc(N * sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid * sizeof(float));	// Output of length 32 or as N based above

	// GPU side memory allocation
	cudaMalloc((void**)&device_a, N * sizeof(float));
	cudaMalloc((void**)&device_b, N * sizeof(float));
	cudaMalloc((void**)&device_partial_c, blocksPerGrid * sizeof(float));

	// Filling host memory with the required data
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	// Copying a and b arrays to the GPU
	cudaMemcpy(device_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

	// Invoking main computation kernel
	dotproduct_gpu <<<blocksPerGrid, threadsPerBlock>>>(device_a, device_b, device_partial_c);

	// Copying the output array from the GPU back to the CPU
	cudaMemcpy(partial_c, device_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

	// Once data is in CPU, finish calculations
	float c = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		c += partial_c[i];
	}

	// Calculating sum of squares for dot product
	#define sumSquares(x) (x*(x+1)*(2*x+1)/6)
	printf("Does GPU value %.6g = %.6g?\n", c, 2 * sumSquares((float)(N - 1)));

	// Memory cleanup and freeing on GPU
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_partial_c);

	// Memory cleanup and freeing on CPU
	free(a);
	free(b);
	free(partial_c);

}
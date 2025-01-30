//#include "book.h"
//#include "./vectoradd.h"
//#include "./juliaset.cuh"	// CPU Juliaset Viz
//#include "./juliaset_gpu.cuh"	// GPU Juliaset Viz
//#include "./dotproduct_gpu.cuh" // Calculation of Dot Product
//#include "./raytracing_gpu.cuh"
//#include "./heattransfer_gpu.cuh"
#include "./atomics_gpu.cuh"

int main()
{
	//Generate_JuliaSet();
	//GPU_Generate_JuliaSet();
	//GPU_CalcDotProduct();
	//GPU_SharedMemoryBitmap();
	//GPU_RayTracing();
	// 
	// Skip Ch7 since CUDA 12.6 does not support "texture" type, rather cudaTextureObject_t. Learn about it separately
	//GPU_HeatTransferMain();

	atomics_gpu_main();
}
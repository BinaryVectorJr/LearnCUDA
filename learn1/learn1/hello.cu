//#include "./vectoradd.h"
//#include "./juliaset.cuh"	// CPU Juliaset Viz
//#include "./juliaset_gpu.cuh"	// GPU Juliaset Viz
//#include "./dotproduct_gpu.cuh" // Calculation of Dot Product
#include "./sharedmembitmap_gpu.cuh"

int main()
{
	//Generate_JuliaSet();
	//GPU_Generate_JuliaSet();
	//GPU_CalcDotProduct();
	GPU_SharedMemoryBitmap();
}
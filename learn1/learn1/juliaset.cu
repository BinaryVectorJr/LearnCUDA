// CPU Implementaton

#include <cpu_bitmap.h>
#include "juliaset.cuh"

#define DIM 1000

/// <summary>
/// Main generation routine (CPU)
/// </summary>
void Generate_JuliaSet()
{
	CPUBitmap uBitmap(DIM, DIM);
	unsigned char* ptr = uBitmap.get_ptr();

	cpu_kernel(ptr);

	uBitmap.display_and_exit();
}

/// <summary>
/// Function to invoke Julia Set on the CPU.
/// </summary>
/// <param name="_ptr">Character pointer to the place where the Bitmap is stored.</param>
void cpu_kernel(unsigned char* _ptr)
{
	for (int y = 0; y < DIM; y++)
	{
		for (int x = 0; x < DIM; x++)
		{
			int offset = x + y * DIM;

			int juliaValue = juliaCalc(x, y);
			_ptr[offset * 4 + 0] = 255 * juliaValue;
			_ptr[offset * 4 + 1] = 255 * juliaValue;
			_ptr[offset * 4 + 2] = 0;
			_ptr[offset * 4 + 3] = 255;
		}
	}
}

/// <summary>
/// Calculation of the juliaset values based on the dimension provided.
/// </summary>
/// <param name="_x">Width</param>
/// <param name="_y">Length</param>
/// <returns></returns>
int juliaCalc(int _x, int _y)
{
	// To zoom in or zoom out
	const float scale = 1.5f;

	// Convert a give (x,y) image point to complex space point at (jx,jy)
	float jx = scale * (float)(DIM / 2 - _x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - _y) / (DIM / 2);

	// Arbitary complex constant based on iterative Julia Set equation
	cuComplex c(-0.8, 0.156);	//-0.8+0.156i
	cuComplex a(jx, jy);

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


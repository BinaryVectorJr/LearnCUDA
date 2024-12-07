// CPU Implementation

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Link external libraries: https://www.programmersought.com/article/5070844086/

void Generate_JuliaSet();
void cpu_kernel(unsigned char* _ptr);
int juliaCalc(int _x, int _y);

/// <summary>
/// Generic structure to store complex numbers
/// </summary>
struct cuComplex
{
	float r;	// Real Component of complex number
	float i;	// Imaginary Component of complex number

	// Complex number constructor
	cuComplex(float _a, float _b) : r(_a), i(_b) {}

	// Calculate the magnitude of a complex number
	float magnitude2(void) { return r * r + i * i; }

	// Defining the '*' operation for complex numbers (MULTIPLICATION)
	cuComplex operator*(const cuComplex& a)
	{
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}

	// Defining the '+' operation for complex numbers (ADDITION)
	cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r + a.r, i + a.i);
	}
};
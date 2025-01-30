#include "./atomics_gpu.cuh"

int atomics_gpu_main(void)
{
	unsigned char* inputBuffer = (unsigned char*)big_random_block(SIZE);	// Input buffer to store random data
	unsigned int mainHistogram[256]; // Create a 256 (8 byte) array to function as bins for each 256 value possible to be stored in 8 bytes
	for (int i = 0; i < 256; i++)
	{
		mainHistogram[i] = 0;	// Initialize the bin array with 0
	}

	// Whenever we see a value "z" in inputBuffer, increment the bin numbered "z" in the mainHistogram
	// If inputBuffer[i] is what we are looking at, example 3, then increment the bin number 3, which is stored in mainHistogram[inputBuffer[i]] i.e. mainHistogram[3] -> which we are colloquially saying is bin number 3 in a series of bin 0 to bin 255
	for (int i = 0; i < SIZE; i++)
	{
		mainHistogram[inputBuffer[i]]++;
	}

	long histoCount = 0;
	for (int i = 0; i < 256; i++)
	{
		histoCount += mainHistogram[i];
	}

	printf("Sum: %1d\n", histoCount);

	free(inputBuffer);
	return 0;
}
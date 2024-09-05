#include <cuda.h>
#include <stdio.h>

__global__ void dot_product(int *A, int *B, int *C, int *n)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < *n)
	{
		C[idx] = A[idx]*B[idx];
	}
}

__global__ void parallel_reduction(int *C, int *n)
{	
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	for (int stride = *n/2; stride >= 1; stride >>= 1)
	{
		if (idx < *n && idx+stride < *n)
		{
			C[idx] += C[idx+stride];
		}
		__syncthreads();
	}
}

int main()
{	
	// NOTE: n should be power of 2 for the parallel reduction to work correctly!

	// create host variables
	int n = 8;
	int A[] = {1, 2, 3, 4, 5, 6, 7, 8};
	int B[] = {4, 3, 2, 1, 8, 7, 6, 5};
	int C[n];

	// create device variables
	int *dev_A, *dev_B, *dev_C, *dev_n;

	// allocate memory for device variables
	cudaMalloc(&dev_A, n*sizeof(int));
	cudaMalloc(&dev_B, n*sizeof(int));
	cudaMalloc(&dev_C, n*sizeof(int));
	cudaMalloc(&dev_n, sizeof(int));

	// tranfer input to device memory
	cudaMemcpy(dev_A, &A, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, &B, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	
	// call kernel and calculate execution time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	dot_product<<<1, n>>>(dev_A, dev_B, dev_C, dev_n);
	parallel_reduction<<<1, n>>>(dev_C, dev_n);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);

	// move result to host variable
	cudaMemcpy(&C, dev_C, n*sizeof(int), cudaMemcpyDeviceToHost);
	
	// display the result
	printf("Parallel Reduction Result: %d\nElapsed Duration: %f\n", C[0], time);
	
	// free memory of device variables
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	cudaFree(dev_n);

	return 0;
}

#include <cuda.h>
#include <stdio.h>

__global__ void odd_even_transposition_sort(int *array, int *n)
{
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	for (int i = 0; i < *n; i++)
	{
		if (i%2) // odd phase
		{	
			if (idx+1 < *n && idx%2 && array[idx] > array[idx+1])
			{
				int temp = array[idx];
				array[idx] = array[idx+1];
				array[idx+1] = temp;
			}
		}
		else	// even phase
		{
			if (idx+1 < *n && idx%2 == 0 && array[idx] > array[idx+1])
			{
				int temp = array[idx+1];
				array[idx+1] = array[idx];
				array[idx] = temp;
			}
		}
		__syncthreads();
	}
}


int main()
{
	// create host variables
	int array[] = {5, 4, 3, 2, 1, 6, 7, 8, 9, 10};
	int n = sizeof(array)/sizeof(array[0]);

	// create device variables
	int *dev_array, *dev_n;

	// allocate device memory
	cudaMalloc(&dev_array, sizeof(int)*n);
	cudaMalloc(&dev_n, sizeof(int));
	
	// copy input to device memory
	cudaMemcpy(dev_array, &array, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice);

	// call kernel and calculate execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        odd_even_transposition_sort<<<1, n>>>(dev_array, dev_n);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
	
	// copy result to host memory
	cudaMemcpy(&array, dev_array, sizeof(int)*n, cudaMemcpyDeviceToHost);

	// display the result
	for (int i = 0; i < n; i++)
	{	
		printf("%d ", array[i]);
	}
	printf("\nExecution Time: %f\n", time);

	// free the device memory
	cudaFree(dev_array);
	cudaFree(dev_n);

	return 0; 
}

#include <cuda.h>
#include <stdio.h>

__constant__ int stencil[] = {1, 2, 3};

__global__ void stencil_compute(int* array, int* result, int *n)
{
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	int left = (idx > 0) ? stencil[0]*array[idx-1] : 0;
	int middle = stencil[1]*array[idx];
	int right = (idx < *n-1) ? stencil[2]*array[idx+1] : 0; 
	result[idx] = left+middle+right;
}

int main()
{	
	// create host variables
	int n = 5;
	int array[] = {1, 2, 3, 4, 5};
	int result[n];
	
	// create device variables
	int *dev_n, *dev_array, *dev_result;
	
	// allocate host memory
	cudaMalloc(&dev_n, sizeof(int));
	cudaMalloc(&dev_array, sizeof(int)*n);
	cudaMalloc(&dev_result, sizeof(int)*n);

	// copy input into device memory
	cudaMemcpy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_array, &array, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_n, &result, sizeof(int)*n, cudaMemcpyHostToDevice);
	
	// call kernel and calculate execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        stencil_compute<<<1, n>>>(dev_array, dev_result, dev_n);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);

	// copy the result to host memory
	cudaMemcpy(&result, dev_result, sizeof(int)*n, cudaMemcpyDeviceToHost);
	
	// display the result
	printf("Array: ");
	for (int i = 0; i < n; i++)
	{	
		printf("%d ", array[i]);
	}
	printf("\nResult: ");
	for (int i = 0; i < n; i++)
	{
		printf("%d ", result[i]);
	}
	printf("\nExecution Time: %f\n", time);

	// free the device variables
	cudaFree(dev_array);
	cudaFree(dev_result);
	cudaFree(dev_n);

	return 0;
}

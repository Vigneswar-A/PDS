#include <cuda.h>
#include <stdio.h>

#define DIM 3

/* 
cost of moving between shared memory < cost of moving between global memory
*/

__global__ void matrix_transpose(int *matrix)
{
	// copy the current element to its correct position on shared matrix
	__shared__ int shared_matrix[DIM][DIM];
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	shared_matrix[idx/DIM][idx%DIM] = matrix[idx];
	
	__syncthreads();
	
	// copy the element to its output location from shared memory
	matrix[idx] = shared_matrix[idx%DIM][idx/DIM];
}

int main()
{
	// create host variables
	int matrix[DIM*DIM] = {1, 2, 3,
			       4, 5, 6,
			       7, 8, 9};

	// create device variables
	int *dev_matrix;

	// allocate device memory
	cudaMalloc(&dev_matrix, sizeof(int)*DIM*DIM);
	cudaMemcpy(dev_matrix, &matrix, sizeof(int)*DIM*DIM, cudaMemcpyHostToDevice);
	
	// display the matrix before transpose
	printf("Before Transpose: \n");
	for (int i = 0; i < DIM*DIM; i++)
	{
		printf("%d ", matrix[i]);
		if ((i+1)%DIM == 0)
		{
			printf("\n");
		}
	}

	// call kernel and calculate execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        matrix_transpose<<<1, DIM*DIM>>>(dev_matrix);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
	
	// copy result to host memory
	cudaMemcpy(&matrix, dev_matrix, sizeof(int)*DIM*DIM, cudaMemcpyDeviceToHost);
	
	// display the result
	printf("After Transpose: \n");
	for (int i = 0; i < DIM*DIM; i++)
	{
		printf("%d ", matrix[i]);
		if ((i+1)%DIM == 0)
		{
			printf("\n");
		}
	}
	printf("Transpose Execution Duration: %f\n", time);
	
	// free device memory
	cudaFree(dev_matrix);
	return 0; 
}





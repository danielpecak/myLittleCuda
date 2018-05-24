#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Matrix multiplication: AxB=C

//CUDA kernel. Each thread takes care of one cell of C matrix
__global__ void matmul(double *a, double *b, double *c, int n)
{
  // Get global thread ID
  int Col = blockIdx.x*blockDim.x+threadIdx.x;
  int Row = blockIdx.y*blockDim.y+threadIdx.y;

  // Not out of bounds
  if((Col<n) && (Row<n)) {// Mutliply matrices
    // c[Row*n + Col] = 0;
    double sum = 0.0;
    for(int k=0;k<n;k++) {
      // c[Row*n + Col] += a[Row*n+k]*b[k*n+Col];
      sum += a[Row*n+k]*b[k*n+Col];
    }
    c[Row*n + Col] = sum;
  }
}

extern "C" void matmul_wrapper(int n, double h_a[], double h_b[], double h_c[])
{
  // Device input matrices
  double *d_a;
  double *d_b;
  // Device output matrices
  double *d_c;

  //Size, in bytes, of each array
  size_t bytes = n*n*sizeof(double);

  // Allocate memory for each matrix on GPU
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  printf(" C Memory allocated \n");

  // Copy host matrices to device
  cudaMemcpy(d_a,h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,h_b, bytes, cudaMemcpyHostToDevice);
  printf(" C Data sent to GPU \n");

  int blockSize, gridSize;
  // Number of threads in each thread block
  blockSize = 32;
  // Number of thread blocks in grid
  gridSize = (int)ceil((double)n/blockSize);

  dim3 dimBlock(blockSize,blockSize);
  dim3 dimGrid(gridSize,gridSize);
  printf("   GridSize: %d\n", gridSize);
  printf("   BlockSize: %d\n", blockSize);

  // Execute the kernel
  matmul<<<dimGrid, dimBlock>>>(d_a,d_b,d_c, n);
  printf(" C Kernel executed \n");

  // Copy array back to host
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // CHECK RESULTS for 3x3 MATRIX
  // printf("%f %f %f\n",h_a[0],h_a[1],h_a[2]);
  // printf("%f %f %f\n",h_a[3],h_a[4],h_a[5]);
  // printf("%f %f %f\n",h_a[6],h_a[7],h_a[8]);
  // printf("\n");
  // printf("%f %f %f\n",h_b[0],h_b[1],h_b[2]);
  // printf("%f %f %f\n",h_b[3],h_b[4],h_b[5]);
  // printf("%f %f %f\n",h_b[6],h_b[7],h_b[8]);
  // printf("\n");
  // printf("%f %f %f\n",h_c[0],h_c[1],h_c[2]);
  // printf("%f %f %f\n",h_c[3],h_c[4],h_c[5]);
  // printf("%f %f %f\n",h_c[6],h_c[7],h_c[8]);

  // Release device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  printf(" C =============== \n");
}

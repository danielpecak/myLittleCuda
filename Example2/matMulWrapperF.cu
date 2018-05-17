#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Matrix multiplication: AxB=C

//CUDA kernel. Each thread takes care of one cell of C matrix
__global__ void matmul(double *a, double *b, double *c, int n)
{
  // Get global thread ID
  int id = ;

  // Not out of bounds
  if(id<n) // Mutliply matrices
    c[id] = 0;
}

extern "C" void matmul_wrapper()
{
  // Size of matrix
  int n = 1000;

  // Host input matrices
  double *h_a;
  double *h_b;
  // Host output matrices
  double *h_c;

  // Device input matrices
  double *d_a;
  double *d_b;
  // Device output matrices
  double *d_c;

  //Size, in bytes, of each array
  size_t bytes = n*n*sizeof(double);

  // Allocate memory for each matrix on host
  h_a = (double*)malloc(bytes);
  h_b = (double*)malloc(bytes);
  h_c = (double*)malloc(bytes);

  // Allocate memory for each matrix on GPU
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Initialize vectors on host
  int i;
  int j;
  for(i=0;i<n;i++) {
    for(j=0;i<n;i++) {
      h_a[i][j] = sinf(i)*sinf(j);
      h_b[i][j] = cosf(i)*cosf(j);
    }
  }
  // Copy host matrices to device
  cudaMemcpy(d_a,h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,h_b, bytes, cudaMemcpyHostToDevice);

  int blockSize, gridSize;

  // Number of threads in each thread block
  blockSize = 1024;
  // Number of thread blocks in grid
  gridSize = (int)ceil((double)n/blockSize);

  // Execute the kernel
  matmul<<<gridSize, blockSize>>>(d_a,d_b,d_c, n);

  // Copy array back to host
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // CHECK RESULTS
  // .................
  // .................
  // .................

  // Release device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Release host memory
  free(h_a);
  free(h_b);
  free(h_c);
}

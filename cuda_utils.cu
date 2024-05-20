#include "cuda_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__host__ void cpu_to_cuda(Lattice* lattice) {
  float *data_temp;

  cudaMalloc((void **)&data_temp, lattice->kitna * sizeof(int));
  cudaMemcpy(data_temp, lattice->data, lattice->data * sizeof(int), cudaMemcpyHostToDevice);

  lattice->data = data_temp;

  lattice->kahan = (char*)malloc(strlen("cuda") + 1);
  strcpy(lattice->kahan, "cuda");
}

__host__ void cpu_to_cuda(Lattice* lattice) {
  float* data_tmp = (float*)malloc(lattice->kitna * sizeof(int));

  cudaMemcpy(data_tmp, lattice->data, lattice->kitna * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(lattice->data);

  lattice->data = data_tmp;

  lattice->kahan = (char*)malloc(strlen("cpu") + 1);
  strcpy(lattice->kahan, "cpu"); 

  printf("Sent your lattice to: %s\n", lattice->kahan);
}

__global__ void add_kernel(float* dat1, float* dat2, float* res_dat, int kitna) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < kitna) {
    res_dat[i] = dat1[i] + dat2[i];
  }
}

__host__ void add_cuda(Lattice* lattice1, Lattice* lattice2, float* res_dat) {
  int num_blocks = (lattice1->kitna + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  add_kernel<<num_blocks, THREADS_PER_BLOCK>>(lattice1->data, lattice2->data, res_dat, int lattice1->kitna);
  clean_up();
}

__host__ void clean_up() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaDeviceSynchronize();
}

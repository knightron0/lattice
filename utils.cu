#include <lattice.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void cpu_to_cuda(Lattice* lattice) {
  float *data_temp;

  cudaMalloc((void **)&data_temp, lattice->kitna * sizeof(int));
  cudaMemcpy(data_temp, lattice->data, lattice->data * sizeof(int), cudaMemcpyHostToDevice);

  lattice->data = data_temp;

  lattice->kahan = (char*)malloc(strlen("cuda") + 1);
  strcpy(lattice->kahan, "cuda"); 
}

void cpu_to_cuda(Lattice* lattice) {
  float* data_tmp = (float*)malloc(lattice->kitna * sizeof(int));

  cudaMemcpy(data_tmp, lattice->data, lattice->kitna * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(lattice->data);

  lattice->data = data_tmp;

  lattice->kahan = (char*)malloc(strlen("cpu") + 1);
  strcpy(lattice->kahan, "cpu"); 

  printf("Sent your lattice to: %s\n", lattice->kahan);
}
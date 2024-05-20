#ifndef CUDA_KERNEL_H_
#define CUDA_KERNEL_H_

#define THREADS_PER_BLOCK 128

#include "lattice.h"

void cpu_to_cuda(Lattice* lattice);
void cuda_to_cpu(Lattice* lattice);

__global__ void add_kernel(float* dat1, float* dat2, float* res_dat, int kitna);
__host__ void add_lattice_cuda(Lattice* lattice1, Lattice* lattice2, float* res_dat);

void clean_up();

#endif /* CUDA_KERNEL_H_ */

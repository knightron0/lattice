#ifndef CUDA_KERNEL_H_
#define CUDA_KERNEL_H_

#include <lattice.h>

#define THREADS_PER_BLOCK 128

__host__ void cpu_to_cuda(Lattice* lattice);
__host__ void cpu_to_cuda(Lattice* lattice);

__global__ void add_kernel(float* dat1, float* dat2, float* res_dat, int kitna);
__host__ void add_cuda(Lattice* lattice1, Lattice* lattice2, float* res_dat);

__host__ void clean_up();

#endif /* CUDA_KERNEL_H_ */
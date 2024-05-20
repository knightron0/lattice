#ifndef CUDA_KERNEL_H_
#define CUDA_KERNEL_H_

#define THREADS_PER_BLOCK 128

void cpu_to_cuda(Lattice* lattice);
void cuda_to_cpu(Lattice* lattice);

#endif /* CUDA_KERNEL_H_ */

#include <lattice.h>

__host__ void cpu_to_cuda(Lattice* lattice) {
  float *kya_temp;

  cudaMalloc(&kya_temp, lattice->kitna * sizeof(int));
  cudaMemcpy(kya_temp, lattice->data, lattice->dat);
}

__host__ void cpu_to_cuda(Lattice* lattice) {

}
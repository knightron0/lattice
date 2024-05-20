#include <lattice.h>

__host__ void cpu_to_cuda(Lattice* lattice) {
  float *kya_temp;

  cudaMalloc(&kya_temp, lattice->kitna * sizeof(int));
  cudaMemcpy(kya_temp, lattice->kya, lattice->kya * sizeof(float));
}

__host__ void cpu_to_cuda(Lattice* lattice) {

}
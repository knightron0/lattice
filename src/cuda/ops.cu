#include "ops.cuh"

__global__ void add_lattice(float *a, float *b, float *c, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < size) c[id] = a[id] + b[id];
}
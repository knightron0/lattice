#include "ops.cuh"

__global__ void add_lattice(float *a, float *b, float *c, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < size) c[id] = a[id] + b[id];
}

__global__ void sub_lattice(float *a, float *b, float *c, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < size) c[id] = a[id] - b[id];
}

__global__ void div_lattice(float *a, float *b, float *c, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < size) c[id] = a[id] / b[id];
}

__global__ void mul_lattice(float *a, float *b, float *c, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < size) c[id] = a[id] * b[id];
}

__global__ void add_scalar_lattice(float *a, float scalar, float *c, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < size) c[id] = a[id] + scalar;
}

__global__ void sub_scalar_lattice(float *a, float scalar, float *c, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < size) c[id] = a[id] - scalar;
}

__global__ void div_scalar_lattice(float *a, float scalar, float *c, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < size) c[id] = a[id] / scalar;
}

__global__ void mul_scalar_lattice(float *a, float scalar, float *c, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < size) c[id] = a[id] * scalar;
}
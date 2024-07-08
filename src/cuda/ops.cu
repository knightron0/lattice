#include <stdio.h>
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

__device__ float gpu_get(float *data, int *stride, int *indices) {
  int index = 0;
  // TOOD: change to ndim
  for (int i = 0; i < 2; i++) index += indices[i] * stride[i];
  return data[index];
}

__device__ void gpu_set(float *data, int *stride, int *indices, float val) {
  int index = 0;
  // TOOD: change to ndim
  for (int i = 0; i < 2; i++) index += indices[i] * stride[i];
  data[index] = val;
}

__global__ void matmul_lattice(float *a, float *b, float *c, int a_rows, int a_cols, int b_cols, int* a_stride, int* b_stride, int* c_stride) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < a_rows && y < b_cols) {
    float tmp = 0.0;
    for (int i = 0; i < a_cols; i++) {
      int a_indices[2] = {x, i};
      int b_indices[2] = {i, y};
      tmp += gpu_get(a, a_stride, a_indices) * gpu_get(b, b_stride, b_indices);
    }
    int c_indices[2] = {x, y};
    gpu_set(c, c_stride, c_indices, tmp);
  }
}
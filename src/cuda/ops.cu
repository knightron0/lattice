#include <stdio.h>
#include "ops.cuh"

/* ------------------ START GPU UTIL FUNCTIONS ------------------*/
__device__ float gpu_get(float *data, int *stride, int *indices, int ndim) {
  int index = 0;
  for (int i = 0; i < 2; i++) index += indices[i] * stride[i];
  return data[index];
}

__device__ void gpu_set(float *data, int *stride, int *indices, float val, int ndim) {
  int index = 0;
  for (int i = 0; i < ndim; i++) index += indices[i] * stride[i];
  data[index] = val;
}
/* ------------------ END GPU UTIL FUNCTIONS ------------------*/

/* ------------------ START ELEMENTWISE OP KERNELS ------------------*/
__global__ void add_lattice(float *a, float *b, float *c, int size, int rows, int cols, int* a_stride, int* b_stride, int* c_stride, int ndim) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < rows && y < cols) {
    int indices[2] = {x, y};
    gpu_set(c, c_stride, indices, gpu_get(a, a_stride, indices, ndim) + gpu_get(b, b_stride, indices, ndim), ndim);
  }
}

__global__ void sub_lattice(float *a, float *b, float *c, int size, int rows, int cols, int* a_stride, int* b_stride, int* c_stride, int ndim) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < rows && y < cols) {
    int indices[2] = {x, y};
    gpu_set(c, c_stride, indices, gpu_get(a, a_stride, indices, ndim) - gpu_get(b, b_stride, indices, ndim), ndim);
  }
}

__global__ void div_lattice(float *a, float *b, float *c, int size, int rows, int cols, int* a_stride, int* b_stride, int* c_stride, int ndim) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < rows && y < cols) {
    int indices[2] = {x, y};
    gpu_set(c, c_stride, indices, gpu_get(a, a_stride, indices, ndim) / gpu_get(b, b_stride, indices, ndim), ndim);
  }
}

__global__ void mul_lattice(float *a, float *b, float *c, int size, int rows, int cols, int* a_stride, int* b_stride, int* c_stride, int ndim) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < rows && y < cols) {
    int indices[2] = {x, y};
    gpu_set(c, c_stride, indices, gpu_get(a, a_stride, indices, ndim) * gpu_get(b, b_stride, indices, ndim), ndim);
  }
}
/* ------------------ END ELEMENTWISE OP KERNELS ------------------*/

/* ------------------ BEGIN SCALAR OP KERNELS ------------------*/
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
/* ------------------ END SCALAR OP KERNELS ------------------*/

/* ------------------ START MATMUL KERNEL ------------------*/
__global__ void matmul_lattice(float *a, float *b, float *c, int a_rows, int a_cols, int b_cols, int* a_stride, int* b_stride, int* c_stride, int a_ndim, int b_ndim, int c_ndim) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < a_rows && y < b_cols) {
    float tmp = 0.0;
    for (int i = 0; i < a_cols; i++) {
      int a_indices[2] = {x, i};
      int b_indices[2] = {i, y};
      tmp += gpu_get(a, a_stride, a_indices, a_ndim) * gpu_get(b, b_stride, b_indices, b_ndim);
    }
    int c_indices[2] = {x, y};
    gpu_set(c, c_stride, c_indices, tmp, c_ndim);
  }
}
/* ------------------ END MATMUL KERNEL ------------------*/

__global__ void sum_lattice(float *a, int size) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < size) {
    __shared__ float partialSum;
    if (threadIdx.x == 0) partialSum = 0.0;
    __syncthreads();
    atomicAdd(&partialSum, a[id]);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(a, partialSum);
  }
}
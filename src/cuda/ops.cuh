#ifndef OPS_CUH
#define OPS_CUH

__global__ void add_lattice(float *a, float *b, float *c, int size, int rows, int cols, int* a_stride, int* b_stride, int* c_stride, int ndim);
__global__ void sub_lattice(float *a, float *b, float *c, int size, int rows, int cols, int* a_stride, int* b_stride, int* c_stride, int ndim);
__global__ void div_lattice(float *a, float *b, float *c, int size, int rows, int cols, int* a_stride, int* b_stride, int* c_stride, int ndim);
__global__ void mul_lattice(float *a, float *b, float *c, int size, int rows, int cols, int* a_stride, int* b_stride, int* c_stride, int ndim);

__global__ void add_scalar_lattice(float *a, float scalar, float *c, int size);
__global__ void sub_scalar_lattice(float *a, float scalar, float *c, int size);
__global__ void div_scalar_lattice(float *a, float scalar, float *c, int size);
__global__ void mul_scalar_lattice(float *a, float scalar, float *c, int size);

__device__ float gpu_get(float *data, int *stride, int *indices, int ndim);
__device__ void gpu_set(float *data, int *stride, int *indices, float val, int ndim);

__global__ void matmul_lattice(float *a, float *b, float *c, int a_rows, int a_cols, int b_cols, int *a_stride, int *b_stride, int *c_stride, int a_ndim, int b_ndim, int c_ndim);

__global__ void sum_lattice(float *a, int size);

#endif /* OPS_CUH */

#ifndef OPS_CUH
#define OPS_CUH

__global__ void add_lattice(float *a, float *b, float *c, int size);
__global__ void sub_lattice(float *a, float *b, float *c, int size);
__global__ void div_lattice(float *a, float *b, float *c, int size);
__global__ void mul_lattice(float *a, float *b, float *c, int size);

__global__ void add_scalar_lattice(float *a, float scalar, float *c, int size);
__global__ void sub_scalar_lattice(float *a, float scalar, float *c, int size);
__global__ void div_scalar_lattice(float *a, float scalar, float *c, int size);
__global__ void mul_scalar_lattice(float *a, float scalar, float *c, int size);

#endif /* OPS_CUH */

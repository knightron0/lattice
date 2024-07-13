#ifndef ACTIVATIONS_CUH
#define ACTIVATIONS_CUH

__global__ void relu_lattice(float *input, float *output, int size);
__global__ void sigmoid_lattice(float *input, float *output, int size);
__global__ void tanh_lattice(float *input, float *output, int size);

#endif /* ACTIVATIONS_CUH */

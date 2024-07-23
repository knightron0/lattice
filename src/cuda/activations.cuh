#ifndef ACTIVATIONS_CUH
#define ACTIVATIONS_CUH

#include "../lattice.cuh"

Lattice relu(const Lattice& input);
Lattice sigmoid(const Lattice& input);
Lattice tanh(const Lattice& input);
Lattice softmax(const Lattice& input);

__global__ void relu_lattice(float *input, float *output, int size);
__global__ void sigmoid_lattice(float *input, float *output, int size);
__global__ void tanh_lattice(float *input, float *output, int size);
__global__ void softmax_lattice(float *input, float *output, int size);

#endif /* ACTIVATIONS_CUH */

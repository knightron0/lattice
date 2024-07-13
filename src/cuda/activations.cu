#include "activations.cuh"

__global__ void relu_lattice(float *input, float *output, int size) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < size) {
    output[id] = fmaxf(0.0, input[id]);
  }
}

__global__ void sigmoid_lattice(float *input, float *output, int size) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < size) {
    output[id] = 1.0 / (1.0 + expf(-input[id]));
  }
}

__global__ void tanh_lattice(float *input, float *output, int size) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < size) {
    output[id] = tanhf(input[id]);
  }
}


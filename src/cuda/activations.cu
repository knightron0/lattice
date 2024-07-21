#include "activations.cuh"

Lattice relu(const Lattice& input) {
  Lattice result = Lattice(input.shapes, input.ndim, ZERO);
  result.send((char *)"cuda");
  relu_lattice<<<ceil((float)input.size / (float) 256), 256>>>(input.data, result.data, input.size);
  return result;
}

Lattice sigmoid(const Lattice& input) {
  Lattice result = Lattice(input.shapes, input.ndim, ZERO);
  result.send((char *)"cuda");
  sigmoid_lattice<<<ceil((float)input.size / (float) 256), 256>>>(input.data, result.data, input.size);
  return result;
}

Lattice tanh(const Lattice& input) {
  Lattice result = Lattice(input.shapes, input.ndim, ZERO);
  result.send((char *)"cuda");
  tanh_lattice<<<ceil((float)input.size / (float) 256), 256>>>(input.data, result.data, input.size);
  return result;
}

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


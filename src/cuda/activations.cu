#include "activations.cuh"
#include "ops.cuh"
#include <stdio.h>
#include <float.h>

#define THREADS_PER_BLOCK 256

Lattice relu(const Lattice& input) {
  Lattice result = Lattice(input.shapes, input.ndim, ZERO);
  result.send((char *)"cuda");
  relu_lattice<<<ceil((float)input.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(input.data, result.data, input.size);
  return result;
}

Lattice sigmoid(const Lattice& input) {
  Lattice result = Lattice(input.shapes, input.ndim, ZERO);
  result.send((char *)"cuda");
  sigmoid_lattice<<<ceil((float)input.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(input.data, result.data, input.size);
  return result;
}

Lattice tanh(const Lattice& input) {
  Lattice result = Lattice(input.shapes, input.ndim, ZERO);
  result.send((char *)"cuda");
  tanh_lattice<<<ceil((float)input.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(input.data, result.data, input.size);
  return result;
}

Lattice softmax(const Lattice& input) {

  Lattice result = Lattice(input.shapes, input.ndim, ZERO);
  result.send((char *)"cuda");
  softmax_lattice<<<ceil((float)input.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(input.data, result.data, input.size);
  result.send((char *)"cpu");
  float result_sum = isinf(result.sum()) ? FLT_MAX : result.sum();
  Lattice final_result = Lattice(input.shapes, input.ndim, ONES);
  final_result.send((char *)"cuda");
  result.send((char *)"cuda");
  div_scalar_lattice<<<ceil((float)result.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(result.data, result_sum, final_result.data, result.size);
  final_result.send((char *)"cpu");
  return final_result;
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

__global__ void softmax_lattice(float *input, float *output, int size) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < size) {
    output[id] = isinf(expf(input[id])) ? FLT_MAX : expf(input[id]);
  }
}
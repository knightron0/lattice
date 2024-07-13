#include "activations.cuh"

Lattice relu(const Lattice& input) {
  Lattice result = Lattice(input.shapes, input.ndim, 0);
  result.send((char *)"cuda");
  relu_lattice<<<ceil((float)input.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(input.data, result.data, input.size);
  return result;
}

Lattice sigmoid(const Lattice& input) {
  Lattice result = Lattice(input.shapes, input.ndim, 0);
  result.send((char *)"cuda");
  sigmoid_lattice<<<ceil((float)input.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(input.data, result.data, input.size);
  return result;
}

Lattice tanh(const Lattice& input) {
  Lattice result = Lattice(input.shapes, input.ndim, 0);
  result.send((char *)"cuda");
  tanh_lattice<<<ceil((float)input.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(input.data, result.data, input.size);
  return result;
}

// Lattice softmax(const Lattice& input) {
//   if (input.ndim != 2) {
//     fprintf(stderr, "Error: Softmax operation requires the lattice to be 2D.\n");
//     exit(1);
//   }
//   Lattice result = Lattice(input.shapes, input.ndim, 0);
//   result.send((char *)"cuda");
//   softmax_lattice<<<input.shapes[0], THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(input.data, result.data, input.shapes[0], input.shapes[1]);
//   return result;
// }


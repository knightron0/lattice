// feedforward neural network for MNIST

#include "../src/lattice.h"
#include <stdio.h>
#include <stdlib.h>

#define BATCH_SIZE 32
#define INPUT_DIM 784
#define HIDDEN1_DIM 256
#define HIDDEN2_DIM 128
#define HIDDEN3_DIM 64
#define OUTPUT_DIM 10


void ff(Lattice* input, Lattice* output, int num_hidden, int* hidden_dims, Lattice** weights, Lattice** biases, Lattice** activations) {
  int batch_size = input->shapes[0];
  int dat_size = batch_size * INPUT_DIM * sizeof(float);
  memcpy(activations[0]->data, input->data, dat_size);

  for (int h = 0; h < num_hidden; h++) {
    int input_dim = (k == 0) ? INPUT_DIM : hidden_dims[k - 1];
    int output_dim = hidden_dims[k];

    // plug in matmul

  }

}

int main(int argc, char **argv) {
  // initialize Lattice** weights, Lattice** biases, Lattice** activations here (hardcoded for now)...
  Lattice* w1 = something( )

  int hidden_dims[] = {HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM};
  int num_hidden = sizeof(hidden_dims) / sizeof(hidden_dims[0]);

  Lattice** weights = (Lattice**)malloc((num_hidden + 1) * sizeof(Lattice*));
  Lattice** biases = (Lattice**)malloc((num_hidden + 1) * sizeof(Lattice*));

  for (int k = 0; k <= num_hidden; k++) {
      // hack to get dims flowing forward properly
      int input_dim = (k == 0) ? INPUT_DIM : hidden_dims[k - 1];
      int output_dim = (k == num_hidden) ? OUTPUT_DIM : hidden_dims[k];

      int weight_shape[] = {input_dim, output_dim};
      weights[k] = rand(weight_shape, 2);

      int bias_shape[] = {1, output_dim};
      biases[k] = rand(bias_shape, 2);
  }

  Lattice** activations = (Lattice**)malloc((num_hidden + 1) * sizeof(Lattice*));
  for (int k = 0; k <= num_hidden; k++) {
      int dim = (k == num_hidden) ? OUTPUT_DIM : hidden_dims[k];
      int activation_shape[] = {BATCH_SIZE, dim};
      activations[k] = crystallize(NULL, activation_shape, 2, "cpu");
  }

  return 0;
}

float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}
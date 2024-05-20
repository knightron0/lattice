// feedforward neural network for MNIST

#include "../src/lattice.h"
#include "losses.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#define BATCH_SIZE 32
#define INPUT_DIM 784
#define HIDDEN1_DIM 256
#define HIDDEN2_DIM 128
#define HIDDEN3_DIM 64
#define OUTPUT_DIM 10

extern "C" {
  Lattice *forward(Lattice* input, int num_hidden, int* hidden_dims, Lattice** weights, Lattice** biases, Lattice** activations) {
    int batch_size = input->shapes[0];
    int dat_size = batch_size * INPUT_DIM * sizeof(float);
    memcpy(activations[0]->data, input->data, dat_size);

    for (int h = 0; h < num_hidden; h++) {
      Lattice* xT_W = matmul(activations[h], weights[h]);
      Lattice* xTW_B = broadcast_add(xT_W, biases[h]);
      // implemenet broadcast for adding bias
      // xT_W += biases[k]

      for (int i = 0; i < xTW_B->kitna; i++) {
        activations[h + 1]->data[i] = relu(xTW_B->data[i]);
      }

      // ? clean up xT_W; or done when goes out of scope
    }
    Lattice* output = matmul(activations[num_hidden], weights[num_hidden]);
    return broadcast_add(output, biases[num_hidden]);
  }

  int main(int argc, char **argv) {
    // initialize Lattice** weights, Lattice** biases, Lattice** activations here (hardcoded for now)...

    int hidden_dims[] = {HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM};
    int num_hidden = sizeof(hidden_dims) / sizeof(hidden_dims[0]);

    Lattice** weights = (Lattice**)malloc((num_hidden + 1) * sizeof(Lattice*));
    Lattice** biases = (Lattice**)malloc((num_hidden + 1) * sizeof(Lattice*));

    for (int k = 0; k <= num_hidden; k++) {
        // hack to get dims flowing forward properly
        int input_dim = (k == 0) ? INPUT_DIM : hidden_dims[k - 1];
        int output_dim = (k == num_hidden) ? OUTPUT_DIM : hidden_dims[k];

        int weight_shape[] = {input_dim, output_dim};
        weights[k] = rand_lattice(weight_shape, 2);

        int bias_shape[] = {1, output_dim};
        biases[k] = rand_lattice(bias_shape, 2);
    }

    Lattice** activations = (Lattice**)malloc((num_hidden + 1) * sizeof(Lattice*));
    for (int k = 0; k <= num_hidden; k++) {
        int dim = (k == num_hidden) ? OUTPUT_DIM : hidden_dims[k];
        int activation_shape[] = {BATCH_SIZE, dim};
        activations[k] = rand_lattice(activation_shape, 2);
    }

    // input shape -> b x 784 for now 
    int input_shape[] = {1, 784};
    Lattice *input = rand_lattice(input_shape, 2);
    Lattice *out = forward(input, num_hidden, hidden_dims, weights, biases, activations);
    printf("shape of out: %d, %d\n", out->shapes[0], out->shapes[1]);
    return 0;
  }
}

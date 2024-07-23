#include "mlp.cuh"
#include <stdio.h>

#define IN_FEATURES 784
#define OUT_FEATURES 10

MLP::MLP(int n_hidden, int* hidden_nodes, ActivationFunction* activations) {
  this->n_layers = n_hidden + 1;
  this->layers = (Linear**) malloc(n_hidden * sizeof(Linear*));
  for (int i = 0; i < this->n_layers; i++) {
    this->layers[i] = new Linear((i == 0) ? IN_FEATURES : hidden_nodes[i-1], (i == this->n_layers - 1) ? OUT_FEATURES : hidden_nodes[i], 1, (char *)"cuda");
  }

  this->activations = activations;
}

Lattice MLP::forward(Lattice x) {
  Lattice result = x;
  for (int i = 0; i < this->n_layers; i++) {
    result = this->layers[i]->forward(result);
    printf("result shape: %d x %d\n", result.shapes[0], result.shapes[1]);
  }
  return result;
}
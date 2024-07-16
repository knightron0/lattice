#include "mlp.cuh"

#define IN_FEATURES 784
#define OUT_FEATURES 10

MLP::MLP(int n_hidden, int* hidden_nodes, ActivationFunction* activations) {
  this->weights = NULL;
}
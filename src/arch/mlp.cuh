#ifndef MLP_CUH
#define MLP_CUH

#include "../lattice.cuh"
#include "linear.cuh"
#include "../cuda/activations.cuh"

enum ActivationFunction {
    RELU = 0,
    SIGMOID = 1,
    TANH = 2,
    SOFTMAX = 3
};

class MLP {
public:
  Linear** layers;
  ActivationFunction* activations;
  int n_layers;
  
  MLP(int n_hidden, int* hidden_nodes, ActivationFunction* activations); 
  Lattice forward(Lattice x);
};

#endif /* MLP_CUH */

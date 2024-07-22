#ifndef MLP_CUH
#define MLP_CUH

#include "../lattice.cuh"
#include "linear.cuh"

enum ActivationFunction {
    RELU,
    SIGMOID,
    TANH,
    SOFTMAX
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

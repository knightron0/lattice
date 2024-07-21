#ifndef MLP_CUH
#define MLP_CUH

#include "../lattice.cuh"

enum ActivationFunction {
    RELU,
    SIGMOID,
    TANH,
    SOFTMAX
};



class MLP {
public:
  Lattice** weights;
  int n_layers;
 

  MLP(int n_hidden, int* hidden_nodes, ActivationFunction* activations); 
};

#endif /* MLP_CUH */

#include "mlp.cuh"

#define IN_FEATURES 784
#define OUT_FEATURES 10

/* IN_FEATURES x hidden_nodes[0]
hidden_nodes[0] x hidden_nodes[1]
.
.
.
hidden_nodes[n_hidden - 1] x OUT_FEATURES
*/


MLP::MLP(int n_hidden, int* hidden_nodes, ActivationFunction* activations) {
  this->n_layers = n_hidden;
  this->weights = (Lattice **)malloc(sizeof(Lattice *) * n_hidden);
  this->activations = 
  for (int i = 0; i < this->n_layers; i++) {
    
  }  
}

Lattice MLP::forward() {

}
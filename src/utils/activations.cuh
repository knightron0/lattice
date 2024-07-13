#ifndef ACTIVATIONS_CUH
#define ACTIVATIONS_CUH

Lattice relu(const Lattice& input);
Lattice sigmoid(const Lattice& input);
Lattice tanh(const Lattice& input);
// Lattice softmax(const Lattice& input);

#endif /* ACTIVATIONS_CUH */
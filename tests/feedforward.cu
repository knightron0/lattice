#include <stdio.h>
#include "../src/arch/linear.cuh"
#include "../src/arch/mlp.cuh"
#include "../src/lattice.cuh"

void printLattice(const Lattice& lattice, const char* name) {
  printf("%s:\n", name);
  int indices[2] = {0, 0};
  for (int i = 0; i < lattice.shapes[0]; i++) {
    for (int j = 0; j < lattice.shapes[1]; j++) {
      indices[0] = i;
      indices[1] = j;
      printf("%f ", lattice.get(indices));
    }
    printf("\n");
  }
}

int main() {
  Linear l1 = Linear(784, 500, 1, (char *)"cuda");
  int x_shape[2] = {1, 784};
  Lattice x = Lattice(x_shape, 2, ONES);
  printLattice(x, "Input Lattice x");

  MLP mlp = MLP(3, new int[3]{500, 64, 32}, new ActivationFunction[4]{RELU, RELU, RELU, SOFTMAX});
  Lattice result = mlp.forward(x);
  printLattice(result, "Output Lattice result");

  return 0;
}
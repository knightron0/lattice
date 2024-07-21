#include <stdio.h>
#include "../src/arch/linear.cuh"
#include "../src/arch/mlp.cuh"

int main() {
  Linear l1 = Linear(784, 500, 1, (char *)"cuda");
  int x_shape[2] = {1, 784};
  Lattice x = Lattice(x_shape, 2, ONES);
  l1.forward(x);
  return 0;
}
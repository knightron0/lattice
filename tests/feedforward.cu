#include <stdio.h>
#include "../src/arch/linear.cuh"
#include "../src/arch/mlp.cuh"

// int main() {
//   Linear l1 = Linear(784, 500, 1, (char *)"cuda");
//   int x_shape[2] = {1, 10};
//   Lattice x = Lattice(x_shape, 2, ONES);
//   MLP mlp = MLP(3, new int[3]{500, 64, 32}, new ActivationFunction[4]{RELU, RELU, RELU, SOFTMAX});
//   mlp.forward(x);
//   return 0;
// }
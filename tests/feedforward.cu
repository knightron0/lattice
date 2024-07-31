#include <stdio.h>
#include "../src/arch/linear.cuh"
#include "../src/arch/mlp.cuh"
#include "../src/lattice.cuh"

int main() {
  int shapes[2] = {3, 3};
  int other_shapes[1] = {1};
  Lattice a = Lattice(shapes, 2, ONES);
  Lattice b = Lattice(other_shapes, 1, ONES); 
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      int a_indices[2] = {i, j};
      a.set(a_indices, i * 3 + j + 1);
    }
  }
  int* res_shapes = broadcast_dim(a, b);
  Lattice a_broadcasted = a.broadcast(res_shapes, 2);
  Lattice b_broadcasted = b.broadcast(res_shapes, 2);
  a_broadcasted.send((char*)"cuda");
  b_broadcasted.send((char*)"cuda");
  Lattice res = a_broadcasted + b_broadcasted;
  res.send((char *)"cpu");
  for (int i= 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      int idx[2] = {i, j};
      printf("%f ", res.get(idx));
    }
    printf("\n");
  }
  

  // Linear l1 = Linear(784, 500, 1, (char *)"cuda");
  // int x_shape[2] = {10, 784};
  // Lattice x = Lattice(x_shape, 2, ONES);
  // MLP mlp = MLP(3, new int[3]{500, 64, 32}, new ActivationFunction[4]{RELU, RELU, RELU, SOFTMAX});
  // Lattice y = mlp.forward(x);
  // y.show(1, 1);
  return 0;
}
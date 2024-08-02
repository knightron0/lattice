#include "linear.cuh"
#include <stdio.h>

Linear::Linear(int in_dim, int out_dim, int bias, char* where) {
  this->in_dim = in_dim;
  this->out_dim = out_dim;
  this->bias = bias;
  
  int w_shape[2] = {out_dim, in_dim};
  this->w = Lattice(w_shape, 2, RANDOM);
  if (this->bias) {
    int b_shape[2] = {1, out_dim};
    this->b = Lattice(b_shape, 2, ZERO);
  }

  this->where = where;
  if (strcmp(this->where, "cuda") == 0) {
    this->send((char *)"cuda");
  }
}

void Linear::send(char *dest) {
  this->w.send(dest);
  if (this->bias) {
    this->b.send(dest);
  }
}

Lattice Linear::forward(Lattice x) {
  if (strcmp(x.where, this->where) != 0) {
    x.send(this->where);
  }
  this->w.T();
  Lattice result = x.matmul(this->w) + this->b;
  printf("[%dx%d] x [%dx%d] + [%dx%d] = [%dx%d]\n", x.shapes[0], x.shapes[1], this->w.shapes[0], this->w.shapes[1], this->b.shapes[0], this->b.shapes[1], result.shapes[0], result.shapes[1]);
  this->w.T();
  return result;
}


#include "linear.cuh"

Linear::Linear(int in_dim, int out_dim, int bias, char* where) {
  this->in_dim = in_dim;
  this->out_dim = out_dim;
  this->bias = bias;
  
  int w_shape[2] = {out_dim, in_dim};
  this->w = Lattice(w_shape, 2, RANDOM);
  if (this->bias) {
    int b_shape[2] = {1, out_dim};
    this->b = Lattice(b_shape, 2, RANDOM);
  }

  this->where = where;
  if (strcmp(this->where, "cuda") == 0) {
    this->send("cuda");
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
  return x.matmul(this->w.T()) + this->b;
}


#include "lattice.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Lattice::Lattice(int *shapes, int ndim) {
  this->shapes = shapes;
  this->ndim = ndim;
  
  this->size = 1;
  for (int i = 0; i < ndim; i++) this->size *= shapes[i];

  this->data = (float*)malloc(this->size * sizeof(float));
  for (int i = 0; i < this->size; i++) this->data[i] = (float)rand() / (float)RAND_MAX;

  this->stride = (int *)malloc(this->ndim * sizeof(int));
  this->stride[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) this->stride[i] = this->stride[i + 1] * this->shapes[i+1];

  this->where = (char *)"cuda";
}

Lattice::~Lattice() {
  free(this->data);
  free(this->shapes);
  free(this->stride);
  free(this->where);
}

float Lattice::get(int *indices) {
  if (sizeof(indices) / sizeof(indices[0]) != this->ndim) {
    printf("Error: Size of indices does not match the number of dimensions in the lattice.\n");
    return 0.0f;
  }
  int index = 0;
  for (int i = 0; i < this->ndim; i++) {
    index += indices[i] * this->stride[i];
  }
  return this->data[index];
}

int main() {
  int shapes[2] = {10, 10};
  int ndim = 2;
  Lattice a = Lattice(shapes, ndim);
  printf("Lattice a: \n");
  int indices[2] = {0, 0};
  for (int i = 0; i < shapes[0]; i++) {
    for (int j = 0; j < shapes[0]; j++) {
      indices[0] = i;
      indices[1] = j;
      printf("%f ", a.get(indices));
    }
    printf("\n");
  }
  Lattice b = Lattice(shapes, ndim);
  printf("Lattice b: \n");
  for (int i = 0; i < shapes[0]; i++) {
    for (int j = 0; j < shapes[0]; j++) {
      indices[0] = i;
      indices[1] = j;
      printf("%f ", b.get(indices));
    }
    printf("\n");
  }
  return 0;
}
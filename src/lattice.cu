#include "lattice.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Lattice* crystallize(int *shapes, int ndim) {
  /*
    shapes: an array of integers representing the shape of the lattice
    ndim: an integer representing the number of dimensions in the lattice

    returns a Lattice with random floats in the range [0, 1]
  */

  Lattice* lattice = (Lattice *)malloc(sizeof(Lattice));
  
  lattice->shapes = shapes;
  lattice->ndim = ndim;
  
  lattice->size = 1;
  for (int i = 0; i < ndim; i++) lattice->size *= shapes[i];

  lattice->data = (float*)malloc(lattice->size * sizeof(float));
  for (int i = 0; i < lattice->size; i++) lattice->data[i] = (float)rand() / (float)RAND_MAX;

  lattice->stride = (int *)malloc(lattice->ndim * sizeof(int));
  lattice->stride[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) lattice->stride[i] = lattice->stride[i + 1] * lattice->shapes[i+1];

  lattice->where = (char *)"cuda";

  return lattice;
}

float get_value(Lattice* lattice, int *indices) {
  if (sizeof(indices) / sizeof(indices[0]) != lattice->ndim) {
    printf("Error: Size of indices does not match the number of dimensions in the lattice.\n");
    return 0.0f;
  }
  int index = 0;
  for (int i = 0; i < lattice->ndim; i++) {
    index += indices[i] * lattice->stride[i];
  }
  return lattice->data[index];
}

int main() {
  int shapes[2] = {10, 10};
  int ndim = 2;
  Lattice* a = crystallize(shapes, ndim);
  printf("Lattice a: \n");
  int indices[2] = {0, 0};
  for (int i = 0; i < shapes[0]; i++) {
    for (int j = 0; j < shapes[0]; j++) {
      indices[0] = i;
      indices[1] = j;
      printf("%f ", get_value(a, indices));
    }
    printf("\n");
  }
  Lattice* b = crystallize(shapes, ndim);
  printf("Lattice b: \n");
  for (int i = 0; i < shapes[0]; i++) {
    for (int j = 0; j < shapes[0]; j++) {
      indices[0] = i;
      indices[1] = j;
      printf("%f ", get_value(b, indices));
    }
    printf("\n");
  }
  return 0;
}
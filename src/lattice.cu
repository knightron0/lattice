#include "lattice.cuh"
#include "cuda/ops.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define THREADS_PER_BLOCK 256

Lattice::Lattice(int *shapes, int ndim, int mode) {
  this->shapes = shapes;
  this->ndim = ndim;
  
  this->size = 1;
  for (int i = 0; i < ndim; i++) this->size *= shapes[i];

  this->data = (float *)malloc(this->size * sizeof(float));
  if (mode == 2) {
    // random
    for (int i = 0; i < this->size; i++) this->data[i] = (float)rand() / (float)RAND_MAX;
  } else if (mode == 1) {
    // ones
    for (int i = 0; i < this->size; i++) this->data[i] = (float)1.0f;
  } else {
    // zero (default)
    for (int i = 0; i < this->size; i++) this->data[i] = (float)0.0f;
  }

  this->stride = (int *)malloc(this->ndim * sizeof(int));
  this->stride[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) this->stride[i] = this->stride[i + 1] * this->shapes[i+1];

  this->where = (char *)"cpu";
}

Lattice::~Lattice() {
  if (strcmp(this->where, "cpu") == 0) {
    free(this->data);
  } else {
    cudaFree(this->data);
  }
  free(this->stride);
}

float Lattice::get(int *indices) {
  if (sizeof(indices) / sizeof(indices[0]) != this->ndim) {
    printf("Error: Size of indices does not match the number of dimensions in the lattice.\n");
    return 0.0f;
  }
  int index = 0;
  for (int i = 0; i < this->ndim; i++) index += indices[i] * this->stride[i];
  return this->data[index];
}

void Lattice::scale(float factor) {
  for (int i = 0; i < this->size; i++) this->data[i] *= factor;
}

void Lattice::to_gpu() {
  float *data_temp;

  cudaMalloc((void **)&data_temp, this->size * sizeof(float));
  cudaMemcpy(data_temp, this->data, this->size * sizeof(float), cudaMemcpyHostToDevice);

  this->data = data_temp;
  this->where = (char *)"cuda"; 
}

void Lattice::to_cpu() {
  float *data_temp = (float*)malloc(this->size * sizeof(float));

  cudaMemcpy(data_temp, this->data, this->size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(this->data);

  this->data = data_temp;
  this->where = (char *)"cpu";
}

void Lattice::send(char *dest) {
  if (dest == this->where) return;
  if (strcmp(dest, "cuda") == 0) {
    this->to_gpu();
  } else {
    this->to_cpu();
  }
}

Lattice Lattice::operator+(const Lattice& other) const {
  if (this->ndim != other.ndim) {
    fprintf(stderr, "Error: Dimensions of lattices do not match.\n");
    exit(1);
  }
  Lattice result = Lattice(this->shapes, this->ndim, 0);
  result.send((char *)"cuda");
  add_lattice<<<ceil((float)this->size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(this->data, other.data, result.data, this->size);
  return result;
}

int main() {
  int shapes[2] = {10, 10};
  int ndim = 2;
  Lattice a = Lattice(shapes, ndim, 2);
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
  Lattice b = Lattice(shapes, ndim, 1);
  printf("Lattice b: \n");
  for (int i = 0; i < shapes[0]; i++) {
    for (int j = 0; j < shapes[0]; j++) {
      indices[0] = i;
      indices[1] = j;
      printf("%f ", b.get(indices));
    }
    printf("\n");
  }
  a.send((char *)"cuda");
  b.send((char *)"cuda");
  Lattice c = a + b;
  c.send((char *)"cpu");
  printf("Lattice c: \n");
  for (int i = 0; i < shapes[0]; i++) {
    for (int j = 0; j < shapes[0]; j++) {
      indices[0] = i;
      indices[1] = j;
      printf("%f ", c.get(indices));
    }
    printf("\n");
  } 
  return 0;
}
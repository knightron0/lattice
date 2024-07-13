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

// Lattice::~Lattice() {
//   if (strcmp(this->where, "cpu") == 0) {
//     free(this->data);
//     free(this->stride);
//   } else {
//     cudaFree(this->data);
//     cudaFree(this->stride);
//   }
// }

float Lattice::get(int *indices) {
  if (sizeof(indices) / sizeof(indices[0]) != this->ndim) {
    printf("Error: Size of indices does not match the number of dimensions in the lattice.\n");
    return 0.0f;
  }
  int index = 0;
  for (int i = 0; i < this->ndim; i++) index += indices[i] * this->stride[i];
  return this->data[index];
}

void Lattice::set(int *indices, float val) {
  if (sizeof(indices) / sizeof(indices[0]) != this->ndim) {
    printf("Error: Size of indices does not match the number of dimensions in the lattice.\n");
    return;
  }
  int index = 0;
  for (int i = 0; i < this->ndim; i++) index += indices[i] * this->stride[i];
  this->data[index] = val;
}

void Lattice::to_gpu() {
  this->where = (char *)"cuda"; 

  float *data_temp;
  cudaMalloc((void **)&data_temp, this->size * sizeof(float));
  cudaMemcpy(data_temp, this->data, this->size * sizeof(float), cudaMemcpyHostToDevice);
  free(this->data);
  this->data = data_temp;
  
  int *stride_temp;
  cudaMalloc((void **)&stride_temp, this->ndim * sizeof(int));
  cudaMemcpy(stride_temp, this->stride, this->ndim * sizeof(int), cudaMemcpyHostToDevice);
  free(this->stride);
  this->stride = stride_temp;
}

void Lattice::to_cpu() {
  this->where = (char *)"cpu";

  float *data_temp = (float*)malloc(this->size * sizeof(float));
  cudaMemcpy(data_temp, this->data, this->size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(this->data);
  this->data = data_temp;
  
  int *stride_temp = (int*)malloc(this->ndim * sizeof(int));
  cudaMemcpy(stride_temp, this->stride, this->ndim * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(this->stride);
  this->stride = stride_temp;
}

void Lattice::send(char *dest) {
  if (dest == this->where) return;
  if (strcmp(dest, "cuda") == 0) {
    this->to_gpu();
  } else {
    this->to_cpu();
  }
}

void Lattice::T() {
  for (int i = 0; i < (this->ndim + 1) / 2; i++) {
    int temp = this->shapes[i];
    this->shapes[i] = this->shapes[this->ndim - 1 - i];
    this->shapes[this->ndim - 1 - i] = temp;

    temp = this->stride[i];
    this->stride[i] = this->stride[this->ndim - 1 - i];
    this->stride[this->ndim - 1 - i] = temp;
  }
}

Lattice Lattice::operator+(const Lattice& other) const {
  if (this->ndim != other.ndim || this->size != other.size) {
    fprintf(stderr, "Error: Dimensions of lattices do not match.\n");
    exit(1);
  }
  for (int i = 0; i < this->ndim; i++) {
    if (this->shapes[i] != other.shapes[i]) {
      fprintf(stderr, "Error: Dimensions of lattices do not match.\n");
      exit(1);
    }
  }
  Lattice result = Lattice(this->shapes, this->ndim, 0);
  result.send((char *)"cuda");
  dim3 gridDim(ceil((float) this->shapes[0] / 32), ceil((float) this->shapes[1] / 32), 1);
  dim3 blockDim(32, 32, 1);
  add_lattice<<<gridDim, blockDim>>>(this->data, other.data, result.data, this->size, this->shapes[0], this->shapes[1], this->stride, other.stride, result.stride, this->ndim);
  return result;
}

Lattice Lattice::operator-(const Lattice& other) const {
  if (this->ndim != other.ndim || this->size != other.size) {
    fprintf(stderr, "Error: Dimensions of lattices do not match.\n");
    exit(1);
  }
  for (int i = 0; i < this->ndim; i++) {
    if (this->shapes[i] != other.shapes[i]) {
      fprintf(stderr, "Error: Dimensions of lattices do not match.\n");
      exit(1);
    }
  }
  Lattice result = Lattice(this->shapes, this->ndim, 0);
  result.send((char *)"cuda");
  dim3 gridDim(ceil((float) this->shapes[0] / 32), ceil((float) this->shapes[1] / 32), 1);
  dim3 blockDim(32, 32, 1);
  sub_lattice<<<gridDim, blockDim>>>(this->data, other.data, result.data, this->size, this->shapes[0], this->shapes[1], this->stride, other.stride, result.stride, this->ndim);
  return result;
}


Lattice Lattice::operator/(const Lattice& other) const {
  if (this->ndim != other.ndim || this->size != other.size) {
    fprintf(stderr, "Error: Dimensions of lattices do not match.\n");
    exit(1);
  }
  for (int i = 0; i < this->ndim; i++) {
    if (this->shapes[i] != other.shapes[i]) {
      fprintf(stderr, "Error: Dimensions of lattices do not match.\n");
      exit(1);
    }
  }
  Lattice result = Lattice(this->shapes, this->ndim, 0);
  result.send((char *)"cuda");
  dim3 gridDim(ceil((float) this->shapes[0] / 32), ceil((float) this->shapes[1] / 32), 1);
  dim3 blockDim(32, 32, 1);
  div_lattice<<<gridDim, blockDim>>>(this->data, other.data, result.data,  this->size, this->shapes[0], this->shapes[1], this->stride, other.stride, result.stride, this->ndim);
  return result;
}

Lattice Lattice::operator*(const Lattice& other) const {
  if (this->ndim != other.ndim || this->size != other.size) {
    fprintf(stderr, "Error: Dimensions of lattices do not match.\n");
    exit(1);
  }
  for (int i = 0; i < this->ndim; i++) {
    if (this->shapes[i] != other.shapes[i]) {
      fprintf(stderr, "Error: Dimensions of lattices do not match.\n");
      exit(1);
    }
  }
  Lattice result = Lattice(this->shapes, this->ndim, 0);
  result.send((char *)"cuda");
  dim3 gridDim(ceil((float) this->shapes[0] / 32), ceil((float) this->shapes[1] / 32), 1);
  dim3 blockDim(32, 32, 1);
  mul_lattice<<<gridDim, blockDim>>>(this->data, other.data, result.data, this->size, this->shapes[0], this->shapes[1], this->stride, other.stride, result.stride, this->ndim);
  return result;
}

template <typename T>
Lattice operator+(const Lattice& lhs, const T& scalar) {
  Lattice result = Lattice(lhs.shapes, lhs.ndim, 0);
  result.send((char *)"cuda");
  add_scalar_lattice<<<ceil((float)lhs.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(lhs.data, (float)scalar, result.data, lhs.size);
  return result; 
}

template <typename T>
Lattice operator-(const Lattice& lhs, const T& scalar) {
  Lattice result = Lattice(lhs.shapes, lhs.ndim, 0);
  result.send((char *)"cuda");
  sub_scalar_lattice<<<ceil((float)lhs.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(lhs.data, (float)scalar, result.data, lhs.size);
  return result; 
}

template <typename T>
Lattice operator/(const Lattice& lhs, const T& scalar) {
  Lattice result = Lattice(lhs.shapes, lhs.ndim, 0);
  result.send((char *)"cuda");
  div_scalar_lattice<<<ceil((float)lhs.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(lhs.data, (float)scalar, result.data, lhs.size);
  return result; 
}

template <typename T>
Lattice operator*(const Lattice& lhs, const T& scalar) {
  Lattice result = Lattice(lhs.shapes, lhs.ndim, 0);
  result.send((char *)"cuda");
  mul_scalar_lattice<<<ceil((float)lhs.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(lhs.data, (float)scalar, result.data, lhs.size);
  return result; 
}

// only works for 2d matrices
Lattice Lattice::matmul(Lattice other) {
  if (this->ndim != 2 || other.ndim != 2) {
    fprintf(stderr, "Error: Matmul operation requires both lattices to be 2D.\n");
    exit(1);
  }
  if (this->shapes[1] != other.shapes[0]) {
    fprintf(stderr, "Error: Incompatible dimensions for matrix multiplication.\n");
    exit(1);
  }

  int new_shapes[2] = {this->shapes[0], other.shapes[1]};
  Lattice result = Lattice(new_shapes, this->ndim, 0);
  result.send((char *)"cuda"); 
  dim3 gridDim(ceil((float) new_shapes[0] / 32), ceil((float) new_shapes[1] / 32), 1);
  dim3 blockDim(32, 32, 1);
  matmul_lattice<<<gridDim, blockDim>>>(this->data, other.data, result.data, this->shapes[0], this->shapes[1], other.shapes[1], this->stride, other.stride, result.stride, this->ndim, other.ndim, result.ndim);
  return result;
}

int main() {
  int shapes[2] = {2, 3};
  int ndim = 2;
  Lattice a = Lattice(shapes, ndim, 2);
  printf("Lattice a: \n");
  int indices[2] = {0, 0};
  for (int i = 0; i < shapes[0]; i++) {
    for (int j = 0; j < shapes[1]; j++) {
      indices[0] = i;
      indices[1] = j;
      printf("%f ", a.get(indices));
    }
    printf("\n");
  }
  a.T();
  printf("Lattice a after tranpose: \n");
  indices[0] = indices[1] = 0;
  for (int i = 0; i < shapes[0]; i++) {
    for (int j = 0; j < shapes[1]; j++) {
      indices[0] = i;
      indices[1] = j;
      printf("%f ", a.get(indices));
    }
    printf("\n");
  }
  int b_shapes[2] = {3, 2}; 
  Lattice b = Lattice(b_shapes, ndim, 2);
  printf("Lattice b: \n");
  for (int i = 0; i < b_shapes[0]; i++) {
    for (int j = 0; j < b_shapes[1]; j++) {
      indices[0] = i;
      indices[1] = j;
      printf("%f ", b.get(indices));
    }
    printf("\n");
  }
  a.send((char *)"cuda");
  b.send((char *)"cuda");
  Lattice c = a * b;
  c.send((char *)"cpu");
  printf("Lattice c: \n");
  for (int i = 0; i < shapes[0]; i++) {
    for (int j = 0; j < shapes[1]; j++) {
      indices[0] = i;
      indices[1] = j;
      printf("%f ", c.get(indices));
    }
    printf("\n");
  } 
  return 0;
}
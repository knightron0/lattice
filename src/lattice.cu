#include "lattice.cuh"
#include "cuda/ops.cuh"
#include "cuda/activations.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define THREADS_PER_BLOCK 256

Lattice::Lattice() {
  this->data = NULL;
}

Lattice::Lattice(int *shapes, int ndim, Mode mode) {
  this->shapes = (int *)malloc(ndim * sizeof(int));
  for (int i = 0; i < ndim; i++) {
    this->shapes[i] = shapes[i];
  }
  this->ndim = ndim;
  
  this->size = 1;
  for (int i = 0; i < ndim; i++) this->size *= shapes[i];

  this->data = (float *)malloc(this->size * sizeof(float));
  switch (mode) {
    case RANDOM:
      // random
      for (int i = 0; i < this->size; i++) this->data[i] = (float)rand() / (float)RAND_MAX - 0.5;
      break;
    case ONES:
      // ones
      for (int i = 0; i < this->size; i++) this->data[i] = (float)1.0f;
      break;
    default:
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
  printf("%d %d\n", sizeof(indices), sizeof(indices[0]));
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
  if (strcmp(dest, this->where) == 0) return;
  if (strcmp(dest, "cuda") == 0) {
    this->to_gpu();
  } else {
    this->to_cpu();
  }
}

void Lattice::show(int shape, int data) {
  int moved = 0;
  if (strcmp(this->where, "cuda") == 0) {
    moved = 1;
    this->to_cpu();
  }
  if (shape) {
    printf("Shape: ");
    for (int i = 0; i < this->ndim; i++) {
      printf("%d ", this->shapes[i]);
    }
    printf("\n");
  }
  if (data) {
    for (int i = 0; i < this->size; i++) {
      printf("%f ", this->data[i]);
    }
    printf("\n");
  }
  if (moved == 1) {
    this->to_gpu();
  }
}

float Lattice::sum() {
  float sum = 0.0;
  if (strcmp(this->where, "cuda") == 0) {
    sum_lattice<<<1, 1>>>(this->data, this->size);
    cudaDeviceSynchronize();
    cudaMemcpy(&sum, this->data, sizeof(float), cudaMemcpyDeviceToHost);
  } else {
    for (int i = 0; i < this->size; i++) {
      sum += this->data[i];
    }
  }
  return sum;
}

void Lattice::T() {
  for (int i = 0; i < (this->ndim + 1) / 2; i++) {
    int temp = this->shapes[i];
    this->shapes[i] = this->shapes[this->ndim - 1 - i];
    this->shapes[this->ndim - 1 - i] = temp;
  }

  if (strcmp(this->where, "cuda") == 0) {
    int* d_stride;
    cudaMalloc((void**)&d_stride, this->ndim * sizeof(int));
    cudaMemcpy(d_stride, this->stride, this->ndim * sizeof(int), cudaMemcpyHostToDevice);

    int* d_temp;
    cudaMalloc((void**)&d_temp, sizeof(int));

    for (int i = 0; i < (this->ndim + 1) / 2; i++) {
      cudaMemcpy(d_temp, &d_stride[i], sizeof(int), cudaMemcpyDeviceToDevice);
      cudaMemcpy(&d_stride[i], &d_stride[this->ndim - 1 - i], sizeof(int), cudaMemcpyDeviceToDevice);
      cudaMemcpy(&d_stride[this->ndim - 1 - i], d_temp, sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(this->stride, d_stride, this->ndim * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_stride);
    cudaFree(d_temp);
  } else {
    for (int i = 0; i < (this->ndim + 1) / 2; i++) {
      int temp = this->stride[i];
      this->stride[i] = this->stride[this->ndim - 1 - i];
      this->stride[this->ndim - 1 - i] = temp;
    }
  }
}

Lattice Lattice::broadcast(int *broadcast_shapes, int broadcast_ndim) {
  Lattice broadcasted_lattice = Lattice(broadcast_shapes, broadcast_ndim, ZERO);
  int* broadcast_stride = (int *)malloc(broadcast_ndim * sizeof(int));
  for (int i = 0; i < this->ndim; i++) {
    broadcast_stride[broadcast_ndim - 1 - i] = this->stride[this->ndim - i - 1];
  }
  for (int i = 0; i < broadcast_ndim; i++) {
    if (i >= this->ndim) {
      broadcast_stride[broadcast_ndim - 1 - i] = 0;
    } else if (this->shapes[this->ndim - 1 - i] != broadcast_shapes[broadcast_ndim - 1 - i]){
      broadcast_stride[broadcast_ndim - 1 - i] = 0; 
    }
  }
  broadcasted_lattice.data = this->data;
  broadcasted_lattice.stride = broadcast_stride;
  return broadcasted_lattice;
}

Lattice Lattice::add_bias(Lattice bias) {
  // TODO: add actual checks here i'm just trusting myself 
  if (this->ndim != bias.ndim) {
    fprintf(stderr, "Error: Dimensions of lattices do not match.\n");
    exit(1);
  }
  Lattice result = Lattice(this->shapes, this->ndim, ZERO);
  result.send((char *)"cuda");
  dim3 gridDim(ceil((float) this->shapes[0] / 32), ceil((float) this->shapes[1] / 32), 1);
  dim3 blockDim(32, 32, 1);
  add_bias_lattice<<<gridDim, blockDim>>>(this->data, bias.data, result.data, this->size, this->shapes[0], this->shapes[1], this->stride, bias.stride, result.stride, this->ndim);
  return result;
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
  Lattice result = Lattice(this->shapes, this->ndim, ZERO);
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
  Lattice result = Lattice(this->shapes, this->ndim, ZERO);
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
  Lattice result = Lattice(this->shapes, this->ndim, ZERO);
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
  Lattice result = Lattice(this->shapes, this->ndim, ZERO);
  result.send((char *)"cuda");
  dim3 gridDim(ceil((float) this->shapes[0] / 32), ceil((float) this->shapes[1] / 32), 1);
  dim3 blockDim(32, 32, 1);
  mul_lattice<<<gridDim, blockDim>>>(this->data, other.data, result.data, this->size, this->shapes[0], this->shapes[1], this->stride, other.stride, result.stride, this->ndim);
  return result;
}

template <typename T>
Lattice operator+(const Lattice& lhs, const T& scalar) {
  Lattice result = Lattice(lhs.shapes, lhs.ndim, ZERO);
  result.send((char *)"cuda");
  add_scalar_lattice<<<ceil((float)lhs.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(lhs.data, (float)scalar, result.data, lhs.size);
  return result; 
}


template <typename T>
Lattice operator-(const Lattice& lhs, const T& scalar) {
  Lattice result = Lattice(lhs.shapes, lhs.ndim, ZERO);
  result.send((char *)"cuda");
  sub_scalar_lattice<<<ceil((float)lhs.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(lhs.data, (float)scalar, result.data, lhs.size);
  return result; 
}

template <typename T>
Lattice operator/(const Lattice& lhs, const T& scalar) {
  Lattice result = Lattice(lhs.shapes, lhs.ndim, ZERO);
  result.send((char *)"cuda");
  div_scalar_lattice<<<ceil((float)lhs.size / (float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(lhs.data, (float)scalar, result.data, lhs.size);
  return result; 
}

template <typename T>
Lattice operator*(const Lattice& lhs, const T& scalar) {
  Lattice result = Lattice(lhs.shapes, lhs.ndim, ZERO);
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
  Lattice result = Lattice(new_shapes, this->ndim, ZERO);
  result.send((char *)"cuda"); 
  dim3 gridDim(ceil((float) new_shapes[0] / 32), ceil((float) new_shapes[1] / 32), 1);
  dim3 blockDim(32, 32, 1);
  matmul_lattice<<<gridDim, blockDim>>>(this->data, other.data, result.data, this->shapes[0], this->shapes[1], other.shapes[1], this->stride, other.stride, result.stride, this->ndim, other.ndim, result.ndim);
  return result;
}

int* broadcast_dim(Lattice a, Lattice b) {
  int res_dim = max(a.ndim, b.ndim);
  int* res_shapes = (int *) malloc(res_dim * sizeof(int)); 
  for (int i = 0; i < res_dim; i++) {
    int a_idx = a.ndim - 1 - i;
    int b_idx = b.ndim - 1 - i;
    if (a_idx >= 0 && b_idx >= 0) {
      // both exist
      if (a.shapes[a_idx] == b.shapes[b_idx]) {
        res_shapes[i] = a.shapes[a_idx];
      } else {
        if (a.shapes[a_idx] != 1 && b.shapes[b_idx] != 1) {
          fprintf(stderr, "Error: Incompatible dimensions for broadcasting.\n");
          exit(1);
        }
        res_shapes[i] = max(a.shapes[a_idx], b.shapes[b_idx]);
      }
    } else if (a_idx >= 0 && b_idx < 0) {
      res_shapes[i] = a.shapes[a_idx];
    } else if (a_idx < 0 && b_idx >= 0) {
      res_shapes[i] = b.shapes[b_idx];
    }
  }
  return res_shapes;
}

// int main() {
//   int shapes[2] = {3, 2};
//   int ndim = 2;
//   Lattice a = Lattice(shapes, ndim, RANDOM);
//   printf("Lattice a: \n");
//   int indices[2] = {0, 0};
//   for (int i = 0; i < shapes[0]; i++) {
//     for (int j = 0; j < shapes[1]; j++) {
//       indices[0] = i;
//       indices[1] = j;
//       printf("%f ", a.get(indices));
//     }
//     printf("\n");
//   }
//   int b_shapes[2] = {3, 1};
//   Lattice b = Lattice(b_shapes, ndim, RANDOM);
//   indices[0] = indices[1] = 0;
//   for (int i = 0; i < b_shapes[0]; i++) {
//     for (int j = 0; j < b_shapes[1]; j++) {
//       indices[0] = i;
//       indices[1] = j;
//       printf("%f ", b.get(indices));
//     }
//     printf("\n");
//   }
//   printf("Lattice C:\n");
//   a.send((char *)"cuda");
//   b.send((char *)"cuda");
//   Lattice c = a.add_bias(b);
//   c.send((char *)"cpu");
//   for (int i = 0; i < shapes[0]; i++) {
//     for (int j = 0; j < shapes[1]; j++) {
//       indices[0] = i;
//       indices[1] = j;
//       printf("%f ", c.get(indices));
//     }
//     printf("\n");
//   }
//   // a.send((char *)"cuda");
//   // Lattice b = softmax(a);
//   // b.send((char *)"cpu");
//   // printf("Lattice a: \n");
//   // for (int i = 0; i < shapes[0]; i++) {
//   //   for (int j = 0; j < shapes[1]; j++) {
//   //     indices[0] = i;
//   //     indices[1] = j;
//   //     printf("%f ", b.get(indices));
//   //   }
//   //   printf("\n");
//   // }
//   // a.T();
//   // printf("Lattice a after tranpose: \n");
//   // indices[0] = indices[1] = 0;
//   // for (int i = 0; i < shapes[0]; i++) {
//   //   for (int j = 0; j < shapes[1]; j++) {
//   //     indices[0] = i;
//   //     indices[1] = j;
//   //     printf("%f ", a.get(indices));
//   //   }
//   //   printf("\n");
//   // }
//   // int b_shapes[2] = {2, 3}; 
//   // for (int i = 0; i < b.shapes[0]; i++) {
//   //   for (int j = 0; j < b.shapes[1]; j++) {
//   //     indices[0] = i;
//   //     indices[1] = j;
//   //     printf("%f ", b.get(indices));
//   //   }
//   //   printf("\n");
//   // }
//   // Lattice c = a * b;
//   // c.send((char *)"cpu");
//   // printf("Lattice c: \n");
//   // for (int i = 0; i < shapes[0]; i++) {
//   //   for (int j = 0; j < shapes[1]; j++) {
//   //     indices[0] = i;
//   //     indices[1] = j;
//   //     printf("%f ", c.get(indices));
//   //   }
//   //   printf("\n");
//   // } 
//   return 0;
// }
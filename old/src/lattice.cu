#include <cuda_runtime_api.h>

#include "lattice.cuh"
#include "cpu/cpu_utils.cuh"
#include "cuda/cuda_utils.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Lattice *crystallize(float* data, int* shapes, int ndim, char* kahan) {
  Lattice *lattice = (Lattice *)malloc(sizeof(Lattice));
  if (lattice == NULL) {
    printf("Lattice could not be allocated\n");
    return NULL;
  }
  lattice->data = data;
  lattice->shapes = shapes;
  lattice->ndim = ndim;

  lattice->size = 1;
  for (int i = 0; i < ndim; i++)
    lattice->size *= lattice->shapes[i];

  lattice->where = kahan;
  int mul = 1;
  lattice->stride = (int*)malloc(ndim * sizeof(int));
  for (int i = ndim - 1; i >= 0; i--) {
    lattice->stride[i] = mul;
    mul *= lattice->shapes[i];
  }
  return lattice;
}

float get(Lattice* lattice, int *indices) {
  int idx = 0;

  return 0.0f;
}

void bhej(Lattice* lattice, char* kahan) {
  if (strcmp(kahan, "cuda") == 0 && strcmp(lattice->where, "cpu") == 0) {
    cpu_to_cuda(lattice);
  } else if (strcmp(kahan, "cpu") == 0 && strcmp(lattice->where, "cuda") == 0) {
    cuda_to_cpu(lattice);
  }
}

Lattice* isomerize(Lattice *lattice, int new_ndim, int* new_shapes) {
  int* shapes = (int*)malloc(new_ndim * sizeof(int));
  if (!shapes) {
    fprintf(stderr, "malloc failed for shapes\n");
    exit(1);
  } else {
    for (int i = 0; i < new_ndim; i++) {
      shapes[i] = new_shapes[i];
    }
  }

  char* kahan = (char*)malloc(strlen(lattice->where) + 1);
  if (!kahan) {
    fprintf(stderr, "malloc failed for device\n");
    exit(1);
  } else {
    strcpy(kahan, lattice->where);
  }

  int kitna = 1;
  for (int i = 0; i < new_ndim; i++) {
    kitna *= shapes[i];
  }

  if (kitna != lattice->size) {
    fprintf(stderr, "size mismatch\n");
    exit(1);
  }

  int* stride = (int*)malloc(new_ndim * sizeof(int));
  stride[new_ndim - 1] = 1;
  for (int i = new_ndim - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * shapes[i + 1]; 
  }

  free(shapes);
  free(stride);

  lattice->shapes = shapes;
  lattice->stride = stride;
  lattice->ndim = new_ndim;

}

Lattice* add(Lattice* lattice1, Lattice* lattice2) {
  if (lattice1->ndim != lattice2->ndim || lattice1->size != lattice2->size) {
    fprintf(stderr, "cannot add, ndims || size doesn't match.\n", lattice1->ndim, lattice2->ndim, ".\t", lattice1->size, lattice2->size);
    exit(1);
  }

  if (strcmp(lattice1->where, lattice2->where) != 0) {
    fprintf(stderr, "cannot add, device (kahan) doesn't match.\n", lattice1->where,
            lattice2->where);
    exit(1);
  }

  for (int i = 0; i < lattice1->ndim; i++) {
    if (lattice1->shapes[i] != lattice2->shapes[i]) {
      fprintf(stderr, "cannot add, shape mismatch\n");
      exit(1);
    }
  }

  char* kahan = (char*)malloc(strlen(lattice1->where) + 1);
  if (!kahan) {
    fprintf(stderr, "malloc failed for device\n");
    exit(1);
  } else {
    strcpy(kahan, lattice1->where);
  }

  int ndim = lattice1->ndim;
  int* shapes = (int*)malloc(ndim * sizeof(int));
  if (!shapes) {
    fprintf(stderr, "malloc failed for addition\n");
    exit(1);
  }

  if (strcmp(lattice1->where, "cuda") == 0) {
    float* res_data;
    cudaMalloc((void**)&res_data, lattice1->size * sizeof(float));
    add_cuda(lattice1, lattice2, res_data);
    Lattice* res_lattice = crystallize(res_data, shapes, ndim, lattice1->where);
    return res_lattice;
  } else {
    float* res_data = (float*)malloc(lattice1->size * sizeof(float));
    if (!res_data) {
      fprintf(stderr, "malloc failed\n");
      exit(1);
    }
    add_cpu(lattice1, lattice2, res_data);
    return crystallize(res_data, shapes, ndim, lattice1->where);
  }
}


Lattice* rand_lattice(int* shapes, int ndim) {
  int kitna = 1;
  for (int i = 0; i < ndim; i++) {
    kitna *= shapes[i];
  }

  float* rand_data = (float*)malloc(kitna * sizeof(float));
  for (int i = 0; i < kitna; i++) {
    rand_data[i] = (float)rand() / RAND_MAX;
  }

  char* kahan = (char*)malloc(4 * sizeof(char));
  strcpy(kahan, "cpu");
  Lattice* randed_lattice = crystallize(rand_data, shapes, ndim, kahan);
  return randed_lattice;
}


Lattice* matmul(Lattice *lattice1, Lattice *lattice2) {
  if (strcmp(lattice1->where, lattice2->where) != 0) {
    fprintf(stderr, "devices not the same\n");
    exit(1);
  }
  if (strcmp(lattice1->where, "cpu") == 0) {
    float* res_data = (float*)malloc(lattice1->size * sizeof(float));
    if (!res_data) {
      fprintf(stderr, "malloc failed\n");
      exit(1);
    }
    int *res_shape = (int *) malloc(2 * sizeof(int));
    res_shape[0] = lattice1->shapes[0]; res_shape[1] = lattice2->shapes[1];
    return crystallize(res_data, res_shape, 2, lattice1->where);
  } else {
    // NOT IMPLEMENTED YET -- need CUDA kernels to do this for me 
    return NULL;
  }
}

Lattice* broadcast_add(Lattice* lattice, Lattice* lattice_col) {
  if (strcmp(lattice->where, lattice_col->where) != 0) {
    fprintf(stderr, "devices not the same\n");
    exit(1);
  }

  int max_ndim = lattice->ndim > lattice_col->ndim ? lattice->ndim : lattice_col->ndim;
  int* broadcasted_shapes = (int*)malloc(max_ndim * sizeof(int));
  if (!broadcasted_shapes) {
    fprintf(stderr, "broadcasted shapes malloc erro\n");
    exit(1);
  }

  for (int i = 0; i < max_ndim; i++) {
      int dim1 = i < lattice->ndim ? lattice->shapes[lattice->ndim - 1 - i] : 1;
      int dim2 = i < lattice_col->ndim ? lattice_col->shapes[lattice_col->ndim - 1 - i] : 1;
      if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
          fprintf(stderr, "imcompatible shapes for broadcasting\n");
          exit(1);
      }
      broadcasted_shapes[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2; // copy voer the 1 to right place
  }
  if (strcmp(lattice->where, "cpu") == 0) {
    float* res_data = (float*)malloc(lattice->size * sizeof(float));
    if (!res_data) {
      fprintf(stderr, "malloc failed\n");
      exit(1);
    }
    broadcast_add_cpu(lattice, lattice_col, broadcasted_shapes, res_data);
    return crystallize(res_data, broadcasted_shapes, max_ndim, lattice->where);
  } else {
    // NOT IMPLEMENTED YET -- need CUDA kernels to do this for me 
    return NULL;
  }

}
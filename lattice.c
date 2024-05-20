#include <math.h>
#include "lattice.h"
#include <stdio.h>
#include <stdlib.h>

Lattice *crystallize(float* data, int* shapes, int ndim, char* kahan) {
  Lattice *lattice = (Lattice *) malloc(sizeof(Lattice));
  if (lattice == NULL) {
    printf("Lattice could not be allocated\n");
    return NULL;
  }
  lattice->data = data;
  lattice->shapes = shapes;
  lattice->ndim = ndim;

  lattice->kitna = 1;
  for (int i = 0; i < ndim; i++)
    lattice->kitna *= lattice->shapes[i];

  lattice->kahan = kahan;
  int mul = 1;
  lattice->stride = malloc(ndim * sizeof(int));
  for (int i = ndim - 1; i >= 0; i--) {
    lattice->stride[i] = mul;
    mul *= lattice->shapes[i];
  }
  return lattice;
}

float get(Lattice* lattice, int *indices) {
  int idx = 0;

}

void bhej(Lattice* lattice, char* kahan) {
  if (strcmp(kahan, "cuda") == 0 && strcmp(lattice->kahan, "cpu") == 0) {
    cpu_to_cuda(lattice);
  } else if (strcmp(kahan, "cpu") == 0 && strcmp(lattice->kahan, "cuda") == 0) {
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

  char* kahan = (char*)malloc(strlen(lattice->kahan) + 1);
  if (!kahan) {
    fprintf(stderr, "malloc failed for device\n");
    exit(1);
  } else {
    strcpy(kahan, lattice->kahan);
  }

  int kitna = 1;
  for (int i = 0; i < new_ndim; i++) {
    kitna *= shapes[i];
  }

  if (kitna != lattice->kitna) {
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
  if (lattice1->ndim != lattice2->ndim || lattice1->kitna != lattice2->kitna) {
    fprintf(stderr, "cannot add, ndims || size doesn't match.\n", lattice1->ndim, lattice2->ndim, ".\t", lattice1->kitna, lattice2->kitna);
    exit(1);
  }

  if (strcmp(lattice1->kahan, lattice2->kahan) != 0) {
    fprintf(stderr, "cannot add, device (kahan) doesn't match.\n", lattice1->kahan,
            lattice2->kahan);
    exit(1);
  }

  for (int i = 0; i < lattice1->ndim; i++) {
    if (lattice1->shapes[i] != lattice2->shapes[i]) {
      fprintf(stderr, "cannot add, shape mismatch\n");
      exit(1);
    }
  }

  char* kahan = (char*)malloc(strlen(lattice1->kahan) + 1);
  if (!kahan) {
    fprintf(stderr, "malloc failed for device\n");
    exit(1);
  } else {
    strcpy(kahan, lattice1->kahan);
  }

  int ndim = lattice1->ndim;
  int* shapes = (int*)malloc(ndim * sizeof(int));
  if (!shapes) {
    fprintf(stderr, "malloc failed for addition\n");
    exit(1);
  }

  if (strcmp(lattice1->kahan, "cuda") == 0) {
    float* added_data;
    cudaMalloc((void**)&added_data, lattice1->kitna * sizeof(float));
    // add_tensor_cuda(lattice1, lattice2, added_data);
    return crystallize(added_data, shapes, ndim, kahan);
  } else {
    float* added_data = (float*)malloc(lattice1->kitna * sizeof(float));
    if (!added_data) {
      fprintf(stderr, "malloc failed\n");
      exit(1);
    }
    // add_tensor_cpu(lattice1, lattice2, added_data);
    return create_tensor(added_data, shapes, ndim, kahan);
  }
}


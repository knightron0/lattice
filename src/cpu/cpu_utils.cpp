#include "cpu_utils.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" {
  void add_cpu(Lattice* lattice1, Lattice* lattice2, float* res_dat) {
    for (int i = 0; i < lattice1->kitna; i++) {
      res_dat[i] = lattice1->data[i] + lattice2->data[i];
    }
  }

  
  void matmul_cpu(Lattice* lattice1, Lattice* lattice2, float* res_dat) {
    if (lattice1->ndim != 2 || lattice2->ndim != 2) {
      fprintf(stderr, "need two dimensional lattices");
      exit(1);
    }
    if (lattice1->shapes[1] != lattice2->shapes[0]) {
      fprintf(stderr, "dimensions do not match up");
      exit(1);
    }
    for (int i = 0; i < lattice1->shapes[0]; i++) {
      for (int j = 0; j < lattice2->shapes[1]; j++) {
        float sum = 0.0;
        for (int ii = 0; ii < lattice2->shapes[0]; ii++) {
          sum += lattice1->data[i * lattice2->shapes[0] + ii] * lattice2->data[ii * lattice2->shapes[1] + j];
        }
        res_dat[i * lattice2->shapes[1] + j] = sum;
      }
    }
  }

  void broadcast_add_cpu(Lattice* lattice, Lattice* lattice_col, int* broadcast_shapes, float* res_dat) {
    int max_ndim = (lattice->ndim > lattice_col->ndim) ? lattice->ndim : lattice_col->ndim;
    int* strides1 = (int*)malloc(max_ndim * sizeof(int));
    int* strides2 = (int*)malloc(max_ndim * sizeof(int));
    if (!strides1 || !strides2) {
        fprintf(stderr, "malloc strides broadcast failed\n");
        exit(1);
    }

    int s1 = 1, s2 = 1;
    for (int i = max_ndim - 1; i >= 0; i--) {
        int dim1 = (i < lattice->ndim) ? lattice->shapes[lattice->ndim + i - max_ndim] : 1;
        int dim2 = (i < lattice_col->ndim) ? lattice_col->shapes[(lattice_col->ndim) + i - max_ndim] : 1;
        strides1[i] = dim1 == broadcast_shapes[i] ? s1 : 0;
        strides2[i] = dim2 == broadcast_shapes[i] ? s2 : 0;
        s1 *= broadcast_shapes[i];
        s2 *= broadcast_shapes[i];
    }

    for (int i = 0; i < lattice->kitna; i++) {
        int j1 = 0, j2 = 0;
        int linear_index = i;
        for (int j = max_ndim - 1; j >= 0; j--) {
            int pos = linear_index % broadcast_shapes[j];
            linear_index /= broadcast_shapes[j];
            if (strides1[j] != 0) {
              j1 += pos * strides1[j];
            }
            if (strides2[j] != 0) {
              j2 += pos * strides2[j];
            }
        }
        res_dat[i] = lattice->data[j1] + lattice_col->data[j2];
    }

    free(strides1);
    free(strides2);
  }

}

#include <math.h>
#include <lattice.h>

Lattice *crystallize(float* data, int* shapes, int ndim, char* kahan) {
  Lattice *lattice = (Lattice *) malloc(sizeof(Lattice));
  lattice->data = data;
  lattice->shapes = shapes;
  lattice->ndim = ndim;

  lattice->kitna = 1;
  for (int i = 0; i < ndim; i++) lattice->kitna *= lattice->shapes[i];

  lattice->kahan = kahan;
  int mul = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    lattice->stride[i] = mul;
    mul *= lattice->shapes[i];
  }
  return lattice;
  
}





void bhej(Lattice* lattice, char* kahan) {
  
}

Lattice* isomerize(Lattice *lattice, int* new_shapes) {
  char *kahan = lattice->kahan;
  if (n)
}





#include <math.h>
#include <lattice.h>

Lattice *crystallize(float *data, int *shapes, int ndim, char *kahan) {
  Lattice *lattice = (Lattice *)malloc(sizeof(Lattice));
  lattice->data = data;
  lattice->shapes = shapes;
  lattice->ndim = ndim;

  lattice->kitna = 1;
  for (int i = 0; i < ndim; i++)
    lattice->kitna *= lattice->shapes[i];

  lattice->kahan = kahan;
  int mul = 1;
  for (int i = ndim - 1; i >= 0; i--)
  {
    lattice->stride[i] = mul;
    mul *= lattice->shapes[i];
  }
  return lattice;
}

void bhej(Lattice *lattice, char *kahan) {
  if (strcmp(kahan, "cuda") == 0 && strcmp(lattice->kahan, "cpu") == 0) {
    cpu_to_cuda(lattice);
  } else if (strcmp(kahan, "cpu") == 0 && strcmp(lattice->kahan, "cuda") == 0) {
    cuda_to_cpu(lattice);
  }
}

Lattice *isomerize(Lattice *lattice, int *new_shapes) {
  char *kahan = lattice->kahan;
  if (n)
}

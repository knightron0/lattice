#include "cpu_utils.h"


extern "C" {
void add_cpu(Lattice* lattice1, Lattice* lattice2, float* res_dat) {
  for (int i = 0; i < lattice1->kitna; i++) {
    res_dat[i] = lattice1->data[i] + lattice2->data[i];
  }
}

void matmul_cpu(Lattice* lattice1, Lattice* lattice2, float* res_dat) {

}
}

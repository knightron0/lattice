#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include "lattice.h"

void add_cpu(Lattice* lattice1, Lattice* lattice2, float* res_dat);
void matmul_cpu(Lattice* lattice1, Lattice* lattice2, float* res_dat);

#endif /* CPU_UTILS_H */
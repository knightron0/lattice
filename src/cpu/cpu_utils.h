#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include "lattice.h"

extern "C" {
	void add_cpu(Lattice* lattice1, Lattice* lattice2, float* res_dat);
	void matmul_cpu(Lattice* lattice1, Lattice* lattice2, float* res_dat);
	void broadcast_add_cpu(Lattice* lattice, Lattice* lattice_col, int* broadcasted_shapes, float* res_dat);
}

#endif /* CPU_UTILS_H */

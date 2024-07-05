#ifndef LATTICE_CUH
#define LATTICE_CUH

typedef struct {
    float *data; // data pointer
    int* shapes; // shape of the lattice
    int* stride; // stride of the lattice
    int ndim; // number of dimensions
    int size; // size of the lattice (= dimension product)
    char* where; // location of the lattice
} Lattice;


Lattice* crystallize(int *shapes, int ndim); 

#endif /* LATTICE_CUH */

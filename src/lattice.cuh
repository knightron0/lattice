#ifndef LATTICE_CUH
#define LATTICE_CUH

class Lattice {
public:
    float *data; // data pointer
    int* shapes; // shape of the lattice
    int* stride; // stride of the lattice
    int ndim; // number of dimensions
    int size; // size of the lattice (= dimension product)
    char* where; // location of the lattice

    Lattice(int *shapes, int ndim);
    ~Lattice();
    float get(int *indices);
};


#endif /* LATTICE_CUH */

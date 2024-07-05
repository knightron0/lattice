#ifndef LATTICE_CUH
#define LATTICE_CUH

typedef struct {
    float *data;
    int* shapes;
    int* stride;
    int ndim;
    int size;
    char* where;
} Lattice;


Lattice* crystallize(float* data, int* shapes, int ndim, char* where); // create 
float get(Lattice *lattice, int *indices); 
Lattice* add(Lattice *lattice1, Lattice *lattice2);
Lattice* scale(Lattice *lattice, float factor);
void send(Lattice* lattice, char* where); // pytorch to_device
Lattice* matmul(Lattice *lattice1, Lattice *lattice2);
Lattice* isomerize(Lattice *lattice, int new_ndim, int* new_shapes); // reshape
Lattice* initialize(int* shapes, int ndim); // 
Lattice* broadcast_add(Lattice* lattice, Lattice* lattice_col);

#endif /* LATTICE_CUH */

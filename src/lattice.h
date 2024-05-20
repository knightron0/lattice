#ifndef LATTICE_H
#define LATTICE_H

typedef struct {
    float *data;
    int* shapes;
    int* stride;
    int ndim;
    int kitna; // size
    char* kahan; // where/device
} Lattice;

Lattice* crystallize(float* data, int* shapes, int ndim, char* kahan); // create 
float get(Lattice *lattice, int *indices); 
Lattice* add(Lattice *lattice1, Lattice *lattice2);
Lattice* scale(Lattice *lattice, float factor);
void bhej(Lattice* lattice, char* kahan); // pytorch to_device
Lattice* matmul(Lattice *lattice1, Lattice *lattice2);
Lattice* isomerize(Lattice *lattice, int new_ndim, int* new_shapes); // reshape
Lattice* rand(int* shapes, int ndim);

#endif /* LATTICE_H */

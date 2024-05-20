// testing file for now
#include <stdio.h>
#include <stdlib.h>
#include "lattice.h"

extern "C" {
int main(int argc, char** argv) {
    float *data = (float*)malloc(10 * sizeof(float));
    for (int i = 0; i < 10; i++) {	
    	data[i] = (float) i;
    }
    int *shapes = (int*)malloc(2 * sizeof(int));
    shapes[0] = 2; shapes[1] = 5;
    Lattice *l = crystallize(data, shapes, 2, "gaosidjf");
    int *idxs = (int *)malloc(2 * sizeof(int));
    idxs[0] = 1; idxs[1] = 4;
    printf("%f\n", get(l, idxs));
    return 0;
}
}

// testing file for now
#include <stdio.h>
#include <stdlib.h>
#include "lattice.h"

int main(int argc, char** argv) {
    float *data = malloc(10 * sizeof(float));
    int *shapes = malloc(2 * sizeof(int));
    shapes[0] = 2; shapes[1] = 5;
    Lattice *l = crystallize(data, shapes, 2, "gaosidjf");

    return 0;
}
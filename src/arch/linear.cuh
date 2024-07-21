#ifndef LINEAR_CUH
#define LINEAR_CUH

#include "../lattice.cuh"

// implements y = xA' + b 
class Linear {
  public: 
    int in_dim;
    int out_dim;
    
    int bias = 1;
    char* where; 

    Lattice w, b;    

    Linear(int in_dim, int out_dim, int bias, char* where);

    Lattice forward(Lattice x);
    void send(char *dest);
     
};

#endif /* LINEAR_CUH */
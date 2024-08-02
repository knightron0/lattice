#ifndef LATTICE_CUH
#define LATTICE_CUH

enum Mode {
  ZERO = 0,
  ONES = 1,
  RANDOM = 2
};

class Lattice {
public:
  float *data; // data pointer
  int* shapes; // shape of the lattice
  int* stride; // stride of the lattice
  int ndim; // number of dimensions
  int size; // size of the lattice (= dimension product)
  char* where; // location of the lattice

  Lattice();
  Lattice(int *shapes, int ndim, Mode mode);
  // ~Lattice();
  float get(int *indices);
  void set(int *indices, float val);
  void send(char *dest);
  void reshape(int *new_shapes, int new_ndim); 
  void to_gpu();
  void to_cpu();
  void show(int shape, int data);
  // a copy() function to another lattice?

  void T();
  float sum();
  Lattice broadcast(int *broadcast_shapes, int broadcast_ndim);

  Lattice operator+(const Lattice& other) const;
  Lattice operator-(const Lattice& other) const;
  Lattice operator*(const Lattice& other) const;
  Lattice operator/(const Lattice& other) const;
  Lattice matmul(Lattice other);

  template <typename T>
  friend Lattice operator+(const Lattice& lhs, const T& scalar);
  template <typename T>
  friend Lattice operator-(const Lattice& lhs, const T& scalar);
  template <typename T>
  friend Lattice operator*(const Lattice& lhs, const T& scalar);
  template <typename T>
  friend Lattice operator/(const Lattice& lhs, const T& scalar);

};

int* broadcast_dim(Lattice a, Lattice b);

#endif /* LATTICE_CUH */

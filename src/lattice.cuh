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

  Lattice(int *shapes, int ndim, int mode);
  ~Lattice();
  float get(int *indices);
  void set(int *indices, float val);
  void send(char *dest);
  void reshape(int *new_shapes, int new_ndim); 
  void to_gpu();
  void to_cpu();

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


#endif /* LATTICE_CUH */

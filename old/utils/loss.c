#include "../src/lattice.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" {
  float relu(float x) {
      return (x > 0.0f) ? x : 0.0f;
  }
}

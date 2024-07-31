NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TESTS_DIR = tests

CUDA_SRC = $(shell find $(SRC_DIR) -name '*.cu')
CUDA_OBJ = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CUDA_SRC))

TARGET = feedforward

all: $(TARGET)

$(TARGET): $(OBJ_DIR)/feedforward.o $(CUDA_OBJ)
    $(NVCC) -g -o $@ $^

$(OBJ_DIR):
    mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/feedforward.o: $(TESTS_DIR)/feedforward.cu | $(OBJ_DIR)
    $(NVCC) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
    $(NVCC) -c -o $@ $<

clean:
    rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/*/*.o $(TARGET)
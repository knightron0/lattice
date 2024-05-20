CC = gcc
CXX = g++
NVCC = nvcc

CFLAGS = -Wall -g
CXXFLAGS = -Wall -g
NVCCFLAGS = -arch=sm_50

SRCS = main.cpp lattice.cpp utils.cu
OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

TARGET = nobodycares

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ -lcudart

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)


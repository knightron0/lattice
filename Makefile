CC = gcc
CXX = g++
NVCC = nvcc

CFLAGS = -Wall -g
CXXFLAGS = -Wall -g

SRCS = 
OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cu=.cu.o)

TARGET = nobodycares

$(TARGET): $(OBJS) $(CU_OBJS)
	$(NVCC) $(OBJS) -o $@ -lcudart -lcuda

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

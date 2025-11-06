NVCC ?= nvcc
CXXFLAGS := -std=c++20 -O3 --compiler-options=-fno-exceptions
GENCODE := -gencode arch=compute_100,code=sm_100 \
           -gencode arch=compute_100,code=compute_100 \
           -gencode arch=compute_103,code=sm_103 \
           -gencode arch=compute_103,code=compute_103 \
           -gencode arch=compute_120,code=sm_120 \
           -gencode arch=compute_120,code=compute_120 \
           -gencode arch=compute_121,code=sm_121 \
           -gencode arch=compute_121,code=compute_121
LDFLAGS :=
TARGET := test

all: $(TARGET)

$(TARGET): test.cu
	$(NVCC) $(CXXFLAGS) $(GENCODE) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

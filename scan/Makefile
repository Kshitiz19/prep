NVCC      ?= nvcc
ARCH      ?= -arch=sm_89
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo $(ARCH) -ltbb

TARGET := scan
SRC    := scan.cu
DEPS   := kernels.cuh

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC) $(DEPS)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

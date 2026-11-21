NVCC      ?= nvcc
ARCH      ?= -arch=sm_89
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo $(ARCH) -ltbb

TARGET := scan
SRC    := scan.cu
DEPS   := kernels.cuh

.PHONY: all clean run

all: $(TARGET)

# WSL libcuda.so.1 is not in the ld.so cache, so cudart reports
# cudaErrorInsufficientDriver. RUNPATH still applies after LD_LIBRARY_PATH.
LDFLAGS ?= -Xlinker --enable-new-dtags -Xlinker -rpath=/usr/lib/wsl/lib

$(TARGET): $(SRC) $(DEPS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(SRC) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

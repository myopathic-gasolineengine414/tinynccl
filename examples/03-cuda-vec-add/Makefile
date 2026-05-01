NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_61

all: vec_add

vec_add: vec_add.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: vec_add
	./vec_add

clean:
	rm -f vec_add

.PHONY: all run clean

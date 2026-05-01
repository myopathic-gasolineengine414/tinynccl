NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_61 -I../../lib/include
LDLIBS = -L../../lib -ltinynccl -libverbs

all: allreduce

allreduce: main.cu ../../lib/libtinynccl.a
	$(NVCC) $(NVCCFLAGS) -o $@ main.cu $(LDLIBS)

../../lib/libtinynccl.a:
	$(MAKE) -C ../../lib

run: allreduce
	./allreduce 0 127.0.0.1 & sleep 0.5; ./allreduce 1 127.0.0.1; wait

clean:
	rm -f allreduce

.PHONY: all run clean

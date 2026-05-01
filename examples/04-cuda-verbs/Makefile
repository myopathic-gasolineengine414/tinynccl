NVCC = nvcc
CC = gcc
NVCCFLAGS = -O2 -arch=sm_61
CFLAGS = -std=c11 -Wall -Wextra -O2
LDLIBS = -libverbs

all: server client

verbs_common.o: verbs_common.c verbs_common.h
	$(CC) $(CFLAGS) -c $< -o $@

server: server.cu verbs_common.o verbs_common.h
	$(NVCC) $(NVCCFLAGS) -o $@ server.cu verbs_common.o $(LDLIBS)

client: client.cu verbs_common.o verbs_common.h
	$(NVCC) $(NVCCFLAGS) -o $@ client.cu verbs_common.o $(LDLIBS)

run: all
	./server & sleep 0.5; ./client; wait

clean:
	rm -f server client verbs_common.o

.PHONY: all run clean

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2
LDLIBS = -libverbs

all: server client

verbs_common.o: verbs_common.c verbs_common.h
	$(CC) $(CFLAGS) -c verbs_common.c -o $@

server: server.c verbs_common.o verbs_common.h
	$(CC) $(CFLAGS) -o $@ server.c verbs_common.o $(LDLIBS)

client: client.c verbs_common.o verbs_common.h
	$(CC) $(CFLAGS) -o $@ client.c verbs_common.o $(LDLIBS)

run: all
	./server & sleep 0.3; ./client; wait

clean:
	rm -f server client verbs_common.o

.PHONY: all run clean

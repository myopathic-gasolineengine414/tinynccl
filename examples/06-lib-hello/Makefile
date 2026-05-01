CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I../../lib/include
LDLIBS = -L../../lib -ltinynccl

all: hello

hello: main.cpp ../../lib/libtinynccl.a
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDLIBS)

../../lib/libtinynccl.a:
	$(MAKE) -C ../../lib

run: hello
	./hello 0 127.0.0.1 & sleep 0.3; ./hello 1 127.0.0.1; wait

clean:
	rm -f hello

.PHONY: all run clean

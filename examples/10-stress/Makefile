CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I../../lib/include
LDLIBS = -L../../lib -ltinynccl -libverbs

all: stress

stress: main.cpp ../../lib/libtinynccl.a
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDLIBS)

../../lib/libtinynccl.a:
	$(MAKE) -C ../../lib

run-tcp: stress
	./stress 0 127.0.0.1 tcp 100 65536 & sleep 0.3; ./stress 1 127.0.0.1 tcp 100 65536; wait

run-verbs: stress
	./stress 0 127.0.0.1 verbs 100 65536 & sleep 0.3; ./stress 1 127.0.0.1 verbs 100 65536; wait

clean:
	rm -f stress

.PHONY: all run-tcp run-verbs clean

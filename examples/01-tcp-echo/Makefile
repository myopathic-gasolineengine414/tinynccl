CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

all: server client

server: server.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

client: client.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: all
	./server & sleep 0.3; ./client; wait

clean:
	rm -f server client

.PHONY: all run clean

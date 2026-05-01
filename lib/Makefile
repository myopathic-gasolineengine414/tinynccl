CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -Iinclude -Isrc

LIB = libtinynccl.a
OBJS = src/tinynccl.o src/transport_tcp.o src/transport_verbs.o

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

src/%.o: src/%.cpp src/transport.h include/tinynccl.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(LIB) $(OBJS)

.PHONY: all clean

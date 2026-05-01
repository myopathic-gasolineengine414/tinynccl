#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace tinynccl {

enum class Backend { TCP, Verbs };
enum class DataType { Float32 };
enum class ReduceOp { Sum };

class Comm {
public:
    static std::unique_ptr<Comm> create(
        int rank,
        int world_size,
        const std::string& peer_host,
        int port,
        Backend backend = Backend::TCP);

    virtual ~Comm() = default;

    virtual int send(const void* buf, size_t bytes) = 0;
    virtual int recv(void* buf, size_t bytes) = 0;

    // In-place all-reduce. Naive 2-rank only for now.
    virtual int all_reduce(
        void* buf,
        size_t count,
        DataType dtype = DataType::Float32,
        ReduceOp op = ReduceOp::Sum) = 0;

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }

protected:
    int rank_ = 0;
    int world_size_ = 0;
};

}

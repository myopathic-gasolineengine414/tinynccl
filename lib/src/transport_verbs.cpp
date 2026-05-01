#include "transport.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <infiniband/verbs.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unordered_map>

namespace tinynccl {

namespace {

constexpr const char* kDeviceName = "rxe0";

struct QpInfo {
    uint32_t qpn;
    uint32_t psn;
    uint16_t lid;
    union ibv_gid gid;
} __attribute__((packed));

class VerbsTransport : public Transport {
public:
    ~VerbsTransport() override {
        for (auto& kv : mr_cache_) ibv_dereg_mr(kv.second.mr);
        mr_cache_.clear();
        if (qp_) ibv_destroy_qp(qp_);
        if (cq_) ibv_destroy_cq(cq_);
        if (pd_) ibv_dealloc_pd(pd_);
        if (ctx_) ibv_close_device(ctx_);
    }

    int establish(int rank, const std::string& peer_host, int port) override {
        if (open_device() != 0) return -1;
        if (create_resources() != 0) return -1;
        if (qp_to_init() != 0) return -1;

        int oob = oob_connect(rank, peer_host, port);
        if (oob < 0) return -1;

        union ibv_gid gid;
        if (ibv_query_gid(ctx_, 1, 0, &gid)) {
            std::fprintf(stderr, "ibv_query_gid failed\n");
            ::close(oob);
            return -1;
        }

        std::srand(static_cast<unsigned>(std::time(nullptr)) + rank);
        QpInfo local = {};
        local.qpn = qp_->qp_num;
        local.psn = static_cast<uint32_t>(std::rand() & 0xffffff);
        local.lid = 0;
        local.gid = gid;

        QpInfo remote = {};
        if (::write(oob, &local, sizeof(local)) != sizeof(local) ||
            ::read(oob, &remote, sizeof(remote)) != sizeof(remote)) {
            std::perror("oob exchange");
            ::close(oob);
            return -1;
        }
        ::close(oob);

        if (qp_to_rtr(remote) != 0) return -1;
        if (qp_to_rts(local.psn) != 0) return -1;
        return 0;
    }

    int send(const void* buf, size_t bytes) override {
        struct ibv_mr* mr = get_or_register(const_cast<void*>(buf), bytes);
        if (!mr) return -1;

        struct ibv_sge sge = {
            reinterpret_cast<uintptr_t>(buf),
            static_cast<uint32_t>(bytes),
            mr->lkey
        };
        struct ibv_send_wr wr = {};
        wr.wr_id = 1;
        wr.opcode = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.send_flags = IBV_SEND_SIGNALED;

        struct ibv_send_wr* bad;
        if (ibv_post_send(qp_, &wr, &bad)) {
            std::perror("post_send");
            return -1;
        }
        return poll_one();
    }

    int recv(void* buf, size_t bytes) override {
        struct ibv_mr* mr = get_or_register(buf, bytes);
        if (!mr) return -1;

        struct ibv_sge sge = {
            reinterpret_cast<uintptr_t>(buf),
            static_cast<uint32_t>(bytes),
            mr->lkey
        };
        struct ibv_recv_wr wr = {};
        wr.wr_id = 1;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        struct ibv_recv_wr* bad;
        if (ibv_post_recv(qp_, &wr, &bad)) {
            std::perror("post_recv");
            return -1;
        }
        return poll_one();
    }

private:
    // Look up (buf,>=bytes) in the MR cache; otherwise register fresh and cache.
    // Cache is keyed by buffer address. Stale entries are not auto-evicted, so
    // callers should reuse buffers across calls (which is the common pattern).
    struct ibv_mr* get_or_register(void* buf, size_t bytes) {
        auto it = mr_cache_.find(buf);
        if (it != mr_cache_.end()) {
            if (it->second.bytes >= bytes) return it->second.mr;
            ibv_dereg_mr(it->second.mr);
            mr_cache_.erase(it);
        }
        struct ibv_mr* mr = ibv_reg_mr(pd_, buf, bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!mr) { std::perror("reg_mr"); return nullptr; }
        mr_cache_[buf] = {mr, bytes};
        return mr;
    }

    int open_device() {
        struct ibv_device** devs = ibv_get_device_list(nullptr);
        if (!devs) { std::perror("ibv_get_device_list"); return -1; }

        struct ibv_device* dev = nullptr;
        for (int i = 0; devs[i]; i++) {
            if (std::strcmp(ibv_get_device_name(devs[i]), kDeviceName) == 0) {
                dev = devs[i];
                break;
            }
        }
        if (!dev) {
            std::fprintf(stderr, "tinynccl: %s not found\n", kDeviceName);
            ibv_free_device_list(devs);
            return -1;
        }

        ctx_ = ibv_open_device(dev);
        ibv_free_device_list(devs);
        return ctx_ ? 0 : -1;
    }

    int create_resources() {
        pd_ = ibv_alloc_pd(ctx_);
        if (!pd_) return -1;
        cq_ = ibv_create_cq(ctx_, 16, nullptr, nullptr, 0);
        if (!cq_) return -1;

        struct ibv_qp_init_attr qa = {};
        qa.send_cq = cq_;
        qa.recv_cq = cq_;
        qa.qp_type = IBV_QPT_RC;
        qa.cap.max_send_wr = 1;
        qa.cap.max_recv_wr = 1;
        qa.cap.max_send_sge = 1;
        qa.cap.max_recv_sge = 1;
        qp_ = ibv_create_qp(pd_, &qa);
        return qp_ ? 0 : -1;
    }

    int qp_to_init() {
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = 1;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;
        int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        return ibv_modify_qp(qp_, &attr, flags);
    }

    int qp_to_rtr(const QpInfo& remote) {
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_1024;
        attr.dest_qp_num = remote.qpn;
        attr.rq_psn = remote.psn;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.dlid = remote.lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = 1;
        attr.ah_attr.grh.dgid = remote.gid;
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.sgid_index = 0;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.traffic_class = 0;
        int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                    IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
        return ibv_modify_qp(qp_, &attr, flags);
    }

    int qp_to_rts(uint32_t my_psn) {
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = my_psn;
        attr.max_rd_atomic = 1;
        int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                    IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
        return ibv_modify_qp(qp_, &attr, flags);
    }

    int oob_connect(int rank, const std::string& peer_host, int port) {
        if (rank == 0) {
            int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (listen_fd < 0) { std::perror("socket"); return -1; }
            int opt = 1;
            ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            sockaddr_in addr = {};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            addr.sin_addr.s_addr = INADDR_ANY;
            if (::bind(listen_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
                std::perror("bind"); ::close(listen_fd); return -1;
            }
            if (::listen(listen_fd, 1) < 0) {
                std::perror("listen"); ::close(listen_fd); return -1;
            }
            int fd = ::accept(listen_fd, nullptr, nullptr);
            ::close(listen_fd);
            return fd;
        } else {
            int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) { std::perror("socket"); return -1; }
            sockaddr_in addr = {};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            ::inet_pton(AF_INET, peer_host.c_str(), &addr.sin_addr);
            for (int retry = 0; retry < 50; ++retry) {
                if (::connect(fd, (sockaddr*)&addr, sizeof(addr)) == 0) return fd;
                if (errno != ECONNREFUSED) { std::perror("connect"); ::close(fd); return -1; }
                ::usleep(100000);
            }
            std::perror("connect (after retries)");
            ::close(fd);
            return -1;
        }
    }

    int poll_one() {
        struct ibv_wc wc;
        int n;
        while ((n = ibv_poll_cq(cq_, 1, &wc)) == 0) {}
        if (n < 0) { std::perror("poll_cq"); return -1; }
        if (wc.status != IBV_WC_SUCCESS) {
            std::fprintf(stderr, "wc: %s\n", ibv_wc_status_str(wc.status));
            return -1;
        }
        return 0;
    }

    struct MrEntry { struct ibv_mr* mr; size_t bytes; };

    struct ibv_context* ctx_ = nullptr;
    struct ibv_pd* pd_ = nullptr;
    struct ibv_cq* cq_ = nullptr;
    struct ibv_qp* qp_ = nullptr;
    std::unordered_map<void*, MrEntry> mr_cache_;
};

}

std::unique_ptr<Transport> make_verbs_transport() {
    return std::make_unique<VerbsTransport>();
}

}

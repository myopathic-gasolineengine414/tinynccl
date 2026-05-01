#ifndef VERBS_COMMON_H
#define VERBS_COMMON_H

#include <stdint.h>
#include <infiniband/verbs.h>

#define DEFAULT_PORT 5000
#define DEVICE_NAME "rxe0"

struct qp_info {
    uint32_t qpn;
    uint32_t psn;
    uint16_t lid;
    union ibv_gid gid;
    uint64_t addr;
    uint32_t rkey;
} __attribute__((packed));

int tcp_listen_accept(int port);
int tcp_connect(const char *host, int port);
int exchange_qp_info(int sock, const struct qp_info *local, struct qp_info *remote);

int qp_to_init(struct ibv_qp *qp, uint8_t port_num);
int qp_to_rtr(struct ibv_qp *qp, uint8_t port_num, const struct qp_info *remote);
int qp_to_rts(struct ibv_qp *qp, uint32_t my_psn);

#endif

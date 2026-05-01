#include "verbs_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <arpa/inet.h>
#include <errno.h>

#define BUF_SIZE 256

static struct ibv_device *find_dev(const char *name) {
    struct ibv_device **list = ibv_get_device_list(NULL);
    for (int i = 0; list && list[i]; i++) {
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) return list[i];
    }
    return NULL;
}

int main(int argc, char **argv) {
    const char *host = (argc > 1) ? argv[1] : "127.0.0.1";
    int port = (argc > 2) ? atoi(argv[2]) : DEFAULT_PORT;
    const char *msg = (argc > 3) ? argv[3] : "hello via rdma write";

    struct ibv_device *dev = find_dev(DEVICE_NAME);
    if (!dev) { fprintf(stderr, "no rxe0\n"); return 1; }

    struct ibv_context *ctx = ibv_open_device(dev);
    struct ibv_pd *pd = ibv_alloc_pd(ctx);

    char *buf = calloc(1, BUF_SIZE);
    strncpy(buf, msg, BUF_SIZE - 1);
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, BUF_SIZE, IBV_ACCESS_LOCAL_WRITE);
    if (!mr) { perror("reg_mr"); return 1; }

    struct ibv_cq *cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);

    struct ibv_qp_init_attr qa = {
        .send_cq = cq, .recv_cq = cq, .qp_type = IBV_QPT_RC,
        .cap = { .max_send_wr = 1, .max_recv_wr = 1,
                 .max_send_sge = 1, .max_recv_sge = 1 },
    };
    struct ibv_qp *qp = ibv_create_qp(pd, &qa);
    if (qp_to_init(qp, 1)) return 1;

    union ibv_gid gid;
    ibv_query_gid(ctx, 1, 0, &gid);

    srand(time(NULL) + 1);
    struct qp_info local = {
        .qpn = qp->qp_num,
        .psn = (uint32_t)(rand() & 0xffffff),
        .lid = 0,
        .gid = gid,
        .addr = 0,
        .rkey = 0,
    };
    struct qp_info remote;

    int sock = tcp_connect(host, port);
    if (sock < 0) return 1;
    if (exchange_qp_info(sock, &local, &remote)) return 1;
    close(sock);

    printf("client: server addr=0x%lx rkey=0x%x\n",
           (unsigned long)remote.addr, remote.rkey);

    if (qp_to_rtr(qp, 1, &remote)) return 1;
    if (qp_to_rts(qp, local.psn)) return 1;

    size_t len = strlen(buf);
    struct ibv_sge sge = { (uintptr_t)buf, len, mr->lkey };
    struct ibv_send_wr swr = {
        .wr_id = 1,
        .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
        .sg_list = &sge,
        .num_sge = 1,
        .send_flags = IBV_SEND_SIGNALED,
        .imm_data = htonl(0xCAFEBABE),
        .wr.rdma.remote_addr = remote.addr,
        .wr.rdma.rkey = remote.rkey,
    };
    struct ibv_send_wr *bad_swr;
    if (ibv_post_send(qp, &swr, &bad_swr)) {
        fprintf(stderr, "post_send: %s\n", strerror(errno));
        return 1;
    }
    printf("client: posted WRITE_WITH_IMM (%zu bytes -> 0x%lx)\n",
           len, (unsigned long)remote.addr);

    struct ibv_wc wc;
    while (ibv_poll_cq(cq, 1, &wc) == 0) {}
    if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "wc: %s\n", ibv_wc_status_str(wc.status));
        return 1;
    }
    printf("client: write completed\n");

    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dereg_mr(mr);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    free(buf);
    return 0;
}

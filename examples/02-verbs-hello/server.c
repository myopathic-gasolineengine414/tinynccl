#include "verbs_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

static struct ibv_device *find_device(const char *name) {
    struct ibv_device **list = ibv_get_device_list(NULL);
    if (!list) return NULL;
    for (int i = 0; list[i]; i++) {
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) return list[i];
    }
    return NULL;
}

int main(int argc, char **argv) {
    int port = (argc > 1) ? atoi(argv[1]) : DEFAULT_PORT;

    struct ibv_device *dev = find_device(DEVICE_NAME);
    if (!dev) { fprintf(stderr, "device %s not found\n", DEVICE_NAME); return 1; }

    struct ibv_context *ctx = ibv_open_device(dev);
    struct ibv_pd *pd = ibv_alloc_pd(ctx);

    char buf[MSG_LEN] = {0};
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, MSG_LEN, IBV_ACCESS_LOCAL_WRITE);

    struct ibv_cq *cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);

    struct ibv_qp_init_attr qp_attr = {
        .send_cq = cq, .recv_cq = cq, .qp_type = IBV_QPT_RC,
        .cap = { .max_send_wr = 1, .max_recv_wr = 1,
                 .max_send_sge = 1, .max_recv_sge = 1 },
    };
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);

    if (qp_to_init(qp, 1)) { fprintf(stderr, "qp_to_init failed\n"); return 1; }

    struct ibv_sge sge = { (uintptr_t)buf, MSG_LEN, mr->lkey };
    struct ibv_recv_wr rwr = { .wr_id = 1, .sg_list = &sge, .num_sge = 1 };
    struct ibv_recv_wr *bad_rwr;
    if (ibv_post_recv(qp, &rwr, &bad_rwr)) {
        fprintf(stderr, "ibv_post_recv failed\n"); return 1;
    }

    union ibv_gid gid;
    if (ibv_query_gid(ctx, 1, 0, &gid)) {
        fprintf(stderr, "ibv_query_gid failed\n"); return 1;
    }

    srand(time(NULL));
    struct qp_info local = {
        .qpn = qp->qp_num,
        .psn = rand() & 0xffffff,
        .lid = 0,
        .gid = gid,
    };
    struct qp_info remote;

    printf("server: rxe0 ready, QPN=%u PSN=%u, awaiting OOB on port %d\n",
           local.qpn, local.psn, port);

    int sock = tcp_listen_accept(port);
    if (sock < 0) return 1;
    if (exchange_qp_info(sock, &local, &remote)) return 1;
    close(sock);

    printf("server: peer QPN=%u PSN=%u\n", remote.qpn, remote.psn);

    if (qp_to_rtr(qp, 1, &remote)) { fprintf(stderr, "qp_to_rtr failed\n"); return 1; }
    if (qp_to_rts(qp, local.psn)) { fprintf(stderr, "qp_to_rts failed\n"); return 1; }

    printf("server: QP in RTS, polling CQ\n");

    struct ibv_wc wc;
    int n;
    while ((n = ibv_poll_cq(cq, 1, &wc)) == 0) {}
    if (n < 0 || wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "wc failed: %s\n", ibv_wc_status_str(wc.status));
        return 1;
    }

    printf("server: received %u bytes via verbs: \"%s\"\n", wc.byte_len, buf);

    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dereg_mr(mr);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    return 0;
}

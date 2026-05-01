#include "verbs_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <arpa/inet.h>

#define BUF_SIZE 256

static struct ibv_device *find_dev(const char *name) {
    struct ibv_device **list = ibv_get_device_list(NULL);
    for (int i = 0; list && list[i]; i++) {
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) return list[i];
    }
    return NULL;
}

int main(int argc, char **argv) {
    int port = (argc > 1) ? atoi(argv[1]) : DEFAULT_PORT;

    struct ibv_device *dev = find_dev(DEVICE_NAME);
    if (!dev) { fprintf(stderr, "no rxe0\n"); return 1; }

    struct ibv_context *ctx = ibv_open_device(dev);
    struct ibv_pd *pd = ibv_alloc_pd(ctx);

    char *buf = calloc(1, BUF_SIZE);
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, BUF_SIZE,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) { perror("reg_mr"); return 1; }

    struct ibv_cq *cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);

    struct ibv_qp_init_attr qa = {
        .send_cq = cq, .recv_cq = cq, .qp_type = IBV_QPT_RC,
        .cap = { .max_send_wr = 1, .max_recv_wr = 1,
                 .max_send_sge = 1, .max_recv_sge = 1 },
    };
    struct ibv_qp *qp = ibv_create_qp(pd, &qa);
    if (qp_to_init(qp, 1)) { fprintf(stderr, "qp_to_init failed\n"); return 1; }

    struct ibv_sge dsg = { (uintptr_t)buf, BUF_SIZE, mr->lkey };
    struct ibv_recv_wr rwr = { .wr_id = 1, .sg_list = &dsg, .num_sge = 1 };
    struct ibv_recv_wr *bad_rwr;
    if (ibv_post_recv(qp, &rwr, &bad_rwr)) { fprintf(stderr, "post_recv failed\n"); return 1; }

    union ibv_gid gid;
    ibv_query_gid(ctx, 1, 0, &gid);

    srand(time(NULL));
    struct qp_info local = {
        .qpn = qp->qp_num,
        .psn = (uint32_t)(rand() & 0xffffff),
        .lid = 0,
        .gid = gid,
        .addr = (uint64_t)(uintptr_t)buf,
        .rkey = mr->rkey,
    };
    struct qp_info remote;

    printf("server: buf=%p rkey=0x%x, awaiting OOB on port %d\n", buf, mr->rkey, port);
    int sock = tcp_listen_accept(port);
    if (sock < 0) return 1;
    if (exchange_qp_info(sock, &local, &remote)) return 1;
    close(sock);

    if (qp_to_rtr(qp, 1, &remote)) { fprintf(stderr, "qp_to_rtr failed\n"); return 1; }
    if (qp_to_rts(qp, local.psn)) { fprintf(stderr, "qp_to_rts failed\n"); return 1; }

    printf("server: polling for write completion\n");
    struct ibv_wc wc;
    while (ibv_poll_cq(cq, 1, &wc) == 0) {}
    if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "wc: %s\n", ibv_wc_status_str(wc.status));
        return 1;
    }
    if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        printf("server: imm=0x%x byte_len=%u\n", ntohl(wc.imm_data), wc.byte_len);
    }
    printf("server: buf contents: \"%s\"\n", buf);

    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dereg_mr(mr);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    free(buf);
    return 0;
}

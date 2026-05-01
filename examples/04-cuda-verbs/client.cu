#include "verbs_common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <ctime>
#include <unistd.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); std::exit(1); \
    } \
} while (0)

#define N_FLOATS 16
#define BUF_BYTES (N_FLOATS * sizeof(float))

__global__ void init_kernel(float *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = (i + 1) * 1.5f;
}

static struct ibv_device *find_device(const char *name) {
    struct ibv_device **list = ibv_get_device_list(NULL);
    if (!list) return NULL;
    for (int i = 0; list[i]; i++) {
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) return list[i];
    }
    return NULL;
}

int main(int argc, char **argv) {
    const char *host = (argc > 1) ? argv[1] : "127.0.0.1";
    int port = (argc > 2) ? atoi(argv[2]) : DEFAULT_PORT;

    float *d_buf;
    CHECK_CUDA(cudaMalloc(&d_buf, BUF_BYTES));

    init_kernel<<<1, N_FLOATS>>>(d_buf, N_FLOATS);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float *h_buf;
    CHECK_CUDA(cudaMallocHost(&h_buf, BUF_BYTES));
    CHECK_CUDA(cudaMemcpy(h_buf, d_buf, BUF_BYTES, cudaMemcpyDeviceToHost));

    printf("client: GPU buffer initialized via kernel: [");
    for (int i = 0; i < N_FLOATS; i++) printf("%s%.1f", i ? "," : "", h_buf[i]);
    printf("]\n");

    struct ibv_device *dev = find_device(DEVICE_NAME);
    if (!dev) { fprintf(stderr, "device %s not found\n", DEVICE_NAME); return 1; }

    struct ibv_context *ctx = ibv_open_device(dev);
    struct ibv_pd *pd = ibv_alloc_pd(ctx);

    struct ibv_mr *mr = ibv_reg_mr(pd, h_buf, BUF_BYTES, IBV_ACCESS_LOCAL_WRITE);
    if (!mr) { perror("ibv_reg_mr"); return 1; }
    printf("client: registered pinned host buffer (lkey=0x%x)\n", mr->lkey);

    struct ibv_cq *cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);

    struct ibv_qp_init_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.send_cq = cq;
    qp_attr.recv_cq = cq;
    qp_attr.qp_type = IBV_QPT_RC;
    qp_attr.cap.max_send_wr = 1;
    qp_attr.cap.max_recv_wr = 1;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);

    if (qp_to_init(qp, 1)) { fprintf(stderr, "qp_to_init\n"); return 1; }

    union ibv_gid gid;
    if (ibv_query_gid(ctx, 1, 0, &gid)) { fprintf(stderr, "query_gid\n"); return 1; }

    srand(time(NULL) + 1);
    struct qp_info local;
    memset(&local, 0, sizeof(local));
    local.qpn = qp->qp_num;
    local.psn = (uint32_t)(rand() & 0xffffff);
    local.lid = 0;
    local.gid = gid;

    struct qp_info remote;

    int sock = tcp_connect(host, port);
    if (sock < 0) return 1;
    if (exchange_qp_info(sock, &local, &remote)) return 1;
    close(sock);

    if (qp_to_rtr(qp, 1, &remote)) { fprintf(stderr, "qp_to_rtr\n"); return 1; }
    if (qp_to_rts(qp, local.psn)) { fprintf(stderr, "qp_to_rts\n"); return 1; }

    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)h_buf;
    sge.length = BUF_BYTES;
    sge.lkey = mr->lkey;

    struct ibv_send_wr swr;
    memset(&swr, 0, sizeof(swr));
    swr.wr_id = 1;
    swr.opcode = IBV_WR_SEND;
    swr.sg_list = &sge;
    swr.num_sge = 1;
    swr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad_swr;
    if (ibv_post_send(qp, &swr, &bad_swr)) { fprintf(stderr, "post_send\n"); return 1; }
    printf("client: posted SEND of %d bytes (data path: GPU -> pinned host -> verbs)\n", (int)BUF_BYTES);

    struct ibv_wc wc;
    int n;
    while ((n = ibv_poll_cq(cq, 1, &wc)) == 0) {}
    if (n < 0 || wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "wc failed: %s\n", ibv_wc_status_str(wc.status));
        return 1;
    }
    printf("client: send completed\n");

    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dereg_mr(mr);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    cudaFreeHost(h_buf);
    cudaFree(d_buf);
    return 0;
}

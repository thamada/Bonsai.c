/*
 * Bonsai GPU forward — CUDA kernels。
 * Q1_0 GEMV: 活性化 Q8_0 化 + vec_dot_q1_0_q8_0（cpu-blas 準拠）。
 * Attention: Flash Attention 風オンライン softmax（K/V タイル走査、att 行列非物質化）。
 */

#include "gpu.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DT_Q1_0 41

/* Flash Attention タイル幅（シーケンス方向）。shared mem ≈ 2×FA_HD×FA_BR floats 以内 */
#define FA_BR  64
#define FA_HD  128   /* Bonsai-8B head_dim */

static __device__ float fa_block_reduce_max(float val)
{
    __shared__ float smem[FA_HD];
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    return smem[0];
}

static __device__ float fa_block_reduce_sum(float val)
{
    __shared__ float smem[FA_HD];
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    return smem[0];
}

/*
 * GQA デコード Attention（1 クエリ位置 × 全ヘッド）。
 * Flash Attention の online softmax で att[npos] を materialize しない。
 * grid: n_heads blocks, block: FA_HD threads。
 */
static __global__ void flash_attn_gqa_kernel(float *xb, const float *q,
    const float *kc, const float *vc, int npos, int n_heads, int hd,
    int kv_dim, int kv_mul, float scale)
{
    int h = blockIdx.x;
    if (h >= n_heads || hd > FA_HD) return;

    int kvh = h / kv_mul;
    const float *qh = q + (size_t)h * hd;
    const float *kbase = kc + (size_t)kvh * hd;
    const float *vbase = vc + (size_t)kvh * hd;
    float *oh = xb + (size_t)h * hd;

    __shared__ float q_sh[FA_HD];
    __shared__ float o_sh[FA_HD];
    __shared__ float scores[FA_BR];

    if (threadIdx.x < hd) {
        q_sh[threadIdx.x] = qh[threadIdx.x];
        o_sh[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;

    for (int t0 = 0; t0 < npos; t0 += FA_BR) {
        int tc = npos - t0;
        if (tc > FA_BR) tc = FA_BR;

        if (threadIdx.x < tc) {
            const float *kt = kbase + (size_t)(t0 + threadIdx.x) * kv_dim;
            float s = 0.0f;
            for (int d = 0; d < hd; d++)
                s += q_sh[d] * kt[d];
            scores[threadIdx.x] = s * scale;
        }
        __syncthreads();

        float m_tile = fa_block_reduce_max(
            (threadIdx.x < tc) ? scores[threadIdx.x] : -1e30f);
        __syncthreads();

        float m_new = fmaxf(m, m_tile);
        float alpha = (m > -1e29f) ? expf(m - m_new) : 0.0f;

        if (threadIdx.x < hd)
            o_sh[threadIdx.x] *= alpha;

        float p_local = 0.0f;
        if (threadIdx.x < tc)
            p_local = expf(scores[threadIdx.x] - m_new);
        float l_tile = fa_block_reduce_sum(p_local);
        __syncthreads();

        if (threadIdx.x < hd) {
            float acc = 0.0f;
            for (int j = 0; j < tc; j++) {
                float p = expf(scores[j] - m_new);
                acc += p * vbase[(size_t)(t0 + j) * kv_dim + threadIdx.x];
            }
            o_sh[threadIdx.x] += acc;
        }
        __syncthreads();

        l = l * alpha + l_tile;
        m = m_new;
    }

    if (threadIdx.x < hd)
        oh[threadIdx.x] = o_sh[threadIdx.x] / l;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static __device__ float dev_f16f32(uint16_t h)
{
    return __half2float(*reinterpret_cast<const __half *>(&h));
}

static __device__ float dev_vec_dot_q1_0_q8_0(int nb,
    const GpuBlockQ1_0 *x, const GpuBlockQ8_0 *y)
{
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d0 = dev_f16f32(x[i].d);
        float sumi = 0.0f;
        for (int k = 0; k < 4; k++) {
            const GpuBlockQ8_0 *yb = &y[i * 4 + k];
            const float d1 = dev_f16f32(yb->d);
            int sumi_block = 0;
            const uint8_t *bits = &x[i].qs[k * 4];
            const int8_t  *qy   = yb->qs;
            for (int b = 0; b < 4; ++b, qy += 8) {
                const unsigned mask = bits[b];
                sumi_block += ((mask & 0x01) ? qy[0] : -qy[0])
                           +  ((mask & 0x02) ? qy[1] : -qy[1])
                           +  ((mask & 0x04) ? qy[2] : -qy[2])
                           +  ((mask & 0x08) ? qy[3] : -qy[3])
                           +  ((mask & 0x10) ? qy[4] : -qy[4])
                           +  ((mask & 0x20) ? qy[5] : -qy[5])
                           +  ((mask & 0x40) ? qy[6] : -qy[6])
                           +  ((mask & 0x80) ? qy[7] : -qy[7]);
            }
            sumi += d1 * (float)sumi_block;
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

static __global__ void quantize_q8_0_kernel(const float *x, GpuBlockQ8_0 *y, int nb)
{
    int ib = blockIdx.x;
    if (ib >= nb) return;

    float amax = 0.0f;
    for (int j = 0; j < GPU_QK8_0; j++) {
        float v = fabsf(x[ib * GPU_QK8_0 + j]);
        if (v > amax) amax = v;
    }
    const float d  = amax / 127.0f;
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;

    y[ib].d = (uint16_t)__half_as_ushort(__float2half_rn(d));
    for (int j = 0; j < GPU_QK8_0; j++)
        y[ib].qs[j] = (int8_t)lrintf(x[ib * GPU_QK8_0 + j] * id);
}

static __global__ void mm_q1_0_kernel(float *o, const GpuBlockQ1_0 *W,
    const GpuBlockQ8_0 *q8, int d, int nb, size_t row_stride)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d) return;
    o[row] = dev_vec_dot_q1_0_q8_0(nb,
        (const GpuBlockQ1_0 *)((const char *)W + row * row_stride), q8);
}

static __global__ void rmsnorm_kernel(float *o, const float *x, const float *w, int n, float eps)
{
    __shared__ float shmem[256];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ss += x[i] * x[i];
    shmem[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(shmem[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        o[i] = x[i] * inv * w[i];
}

static __global__ void rmsnorm_head_kernel(float *vec, const float *w,
    int n_heads, int hd, float eps)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;
    float *seg = vec + h * hd;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x)
        ss += seg[i] * seg[i];
    __shared__ float shmem[256];
    shmem[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(shmem[0] / (float)hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x)
        seg[i] *= inv * w[i];
}

static __global__ void rope_neox_kernel(float *vec, const float *cache,
    int n_heads, int head_dim, int n_rot)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;
    float *row = vec + h * head_dim;
    int nhalf = head_dim / 2;
    int npairs = n_rot / 2;
    for (int j = threadIdx.x; j < nhalf && j < npairs; j += blockDim.x) {
        int i0 = 2 * j;
        float c = cache[i0], s = cache[i0 + 1];
        float x0 = row[j];
        float x1 = row[j + nhalf];
        row[j]         = x0 * c - x1 * s;
        row[j + nhalf] = x0 * s + x1 * c;
    }
}

static __global__ void swiglu_kernel(float *hb, const float *hb2, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float val = hb[i];
    val = val / (1.0f + expf(-val));
    hb[i] = val * hb2[i];
}

static __global__ void add_kernel(float *a, const float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

static __global__ void emb_q1_0_kernel(float *o, const GpuBlockQ1_0 *row, int nb)
{
    int ib = blockIdx.x;
    if (ib >= nb) return;
    const float d = dev_f16f32(row[ib].d);
    const float neg_d = -d;
    for (int j = threadIdx.x; j < GPU_QK1_0; j += blockDim.x) {
        int byte_index = j / 8;
        int bit_offset = j % 8;
        uint8_t bit = (uint8_t)((row[ib].qs[byte_index] >> bit_offset) & 1);
        o[ib * GPU_QK1_0 + j] = bit ? d : neg_d;
    }
}

typedef struct {
    void  *ptr;
    size_t bytes;
} DevBuf;

typedef struct {
    void  **layer;
    size_t *bytes;
    int    n_layers;
} DevLayerBuf;

struct GpuModel {
    GpuConfig cfg;

    DevBuf embd;
    int embd_t;

    DevLayerBuf norm_att, q_norm, k_norm, norm_ffn;
    DevLayerBuf wq, wk, wv, wo, gate, up, down;
    int *wq_t, *wk_t, *wv_t, *wo_t, *gate_t, *up_t, *down_t;

    DevBuf norm_out;
    DevBuf out;
    int out_t;

    float *x, *xb, *xb2, *hb, *hb2;
    float *q, *k, *v, *logits;
    float *kc, *vc;
    GpuBlockQ8_0 *q8;
    int q8_nb;

    float *rope_cache;
    float *h_rope;
};

static DevBuf dev_upload(const void *host, size_t bytes)
{
    DevBuf b = { NULL, bytes };
    if (bytes == 0) return b;
    CUDA_CHECK(cudaMalloc(&b.ptr, bytes));
    CUDA_CHECK(cudaMemcpy(b.ptr, host, bytes, cudaMemcpyHostToDevice));
    return b;
}

static size_t q1_0_tensor_bytes(int n_in, int n_out)
{
    int nb = n_in / GPU_QK1_0;
    return (size_t)n_out * (size_t)nb * sizeof(GpuBlockQ1_0);
}

static DevLayerBuf dev_upload_f32_layers(int n_layers, float * const *host_ptrs, int n_elems)
{
    DevLayerBuf lb = { NULL, NULL, n_layers };
    size_t bytes = (size_t)n_elems * sizeof(float);
    lb.layer = (void **)calloc((size_t)n_layers, sizeof(void *));
    lb.bytes = (size_t *)calloc((size_t)n_layers, sizeof(size_t));
    for (int l = 0; l < n_layers; l++) {
        lb.bytes[l] = bytes;
        CUDA_CHECK(cudaMalloc(&lb.layer[l], bytes));
        CUDA_CHECK(cudaMemcpy(lb.layer[l], host_ptrs[l], bytes, cudaMemcpyHostToDevice));
    }
    return lb;
}

static DevLayerBuf dev_upload_q1_layers(int n_layers, const void * const *host_ptrs,
    int n_in, int n_out)
{
    DevLayerBuf lb = { NULL, NULL, n_layers };
    size_t bytes = q1_0_tensor_bytes(n_in, n_out);
    lb.layer = (void **)calloc((size_t)n_layers, sizeof(void *));
    lb.bytes = (size_t *)calloc((size_t)n_layers, sizeof(size_t));
    for (int l = 0; l < n_layers; l++) {
        lb.bytes[l] = bytes;
        CUDA_CHECK(cudaMalloc(&lb.layer[l], bytes));
        CUDA_CHECK(cudaMemcpy(lb.layer[l], host_ptrs[l], bytes, cudaMemcpyHostToDevice));
    }
    return lb;
}

static void dev_free(DevBuf *b)
{
    if (b && b->ptr) cudaFree(b->ptr);
    if (b) { b->ptr = NULL; b->bytes = 0; }
}

static void dev_free_layers(DevLayerBuf *lb)
{
    if (!lb) return;
    for (int l = 0; l < lb->n_layers; l++)
        if (lb->layer && lb->layer[l]) cudaFree(lb->layer[l]);
    free(lb->layer);
    free(lb->bytes);
    lb->layer = NULL;
    lb->bytes = NULL;
    lb->n_layers = 0;
}

static void gpu_mm_q1_0(GpuModel *gm, float *o, const float *x, const void *W, int n, int d)
{
    int nb = n / GPU_QK1_0;
    int q8_nb = n / GPU_QK8_0;
    size_t row_stride = (size_t)nb * sizeof(GpuBlockQ1_0);

    quantize_q8_0_kernel<<<q8_nb, 128>>>(x, gm->q8, q8_nb);
    mm_q1_0_kernel<<<(d + 255) / 256, 256>>>(o, (const GpuBlockQ1_0 *)W, gm->q8, d, nb, row_stride);
}

static void gpu_mm(GpuModel *gm, float *o, const float *x, const void *W, int n, int d, int type)
{
    if (type == DT_Q1_0) {
        gpu_mm_q1_0(gm, o, x, W, n, d);
        return;
    }
    fprintf(stderr, "gpu_mm: unsupported type %d (Bonsai Q1_0 expected)\n", type);
    exit(1);
}

static void build_rope_cache_host(const GpuConfig *c, int pos, float *cache)
{
    int n_rot = c->n_rot > 0 ? c->n_rot : c->head_dim;

    float corr_lo, corr_hi;
    {
        float beta_fast = c->yarn_beta_fast;
        float beta_slow = c->yarn_beta_slow;
        float n_ctx = (float)c->n_ctx_orig_yarn;
        float start = floorf((float)n_rot * logf(n_ctx / (beta_fast * 2.0f * (float)M_PI))
            / (2.0f * logf(c->rope_theta)));
        float end = ceilf((float)n_rot * logf(n_ctx / (beta_slow * 2.0f * (float)M_PI))
            / (2.0f * logf(c->rope_theta)));
        corr_lo = start > 0.0f ? start : 0.0f;
        corr_hi = end < (float)(n_rot - 1) ? end : (float)(n_rot - 1);
    }

    float theta_scale = powf(c->rope_theta, -2.0f / (float)n_rot);
    float th = (float)pos;

    for (int i0 = 0; i0 < n_rot; i0 += 2) {
        float theta_extrap = th;
        float theta_interp = c->rope_freq_scale * theta_extrap;
        float theta = theta_interp;
        float ms = c->yarn_attn_factor;
        if (c->yarn_ext_factor != 0.0f) {
            float span = (corr_hi - corr_lo) > 0.001f ? (corr_hi - corr_lo) : 0.001f;
            float y = ((float)i0 / 2.0f - corr_lo) / span;
            if (y < 0.0f) y = 0.0f;
            else if (y > 1.0f) y = 1.0f;
            float ramp_mix = (1.0f - y) * c->yarn_ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            ms *= (1.0f + 0.1f * logf(1.0f / c->rope_freq_scale));
        }
        cache[i0]     = cosf(theta) * ms;
        cache[i0 + 1] = sinf(theta) * ms;
        th *= theta_scale;
    }
}

GpuModel *gpu_model_create(const GpuConfig *cfg, const GpuWeightsHost *host)
{
    GpuModel *gm = (GpuModel *)calloc(1, sizeof(GpuModel));
    gm->cfg = *cfg;

    const int L = cfg->n_layers;
    const int dim = cfg->dim;
    const int hidden = cfg->hidden_dim;
    const int kv_dim = cfg->kv_dim;
    const int vocab = cfg->vocab_size;
    const int max_seq = cfg->max_seq;
    const int hd = cfg->head_dim;
    const int n_rot = cfg->n_rot > 0 ? cfg->n_rot : hd;

    gm->embd_t = host->embd_t;
    gm->embd = dev_upload(host->embd, q1_0_tensor_bytes(dim, vocab));

    gm->wq_t = (int *)malloc((size_t)L * sizeof(int));
    gm->wk_t = (int *)malloc((size_t)L * sizeof(int));
    gm->wv_t = (int *)malloc((size_t)L * sizeof(int));
    gm->wo_t = (int *)malloc((size_t)L * sizeof(int));
    gm->gate_t = (int *)malloc((size_t)L * sizeof(int));
    gm->up_t = (int *)malloc((size_t)L * sizeof(int));
    gm->down_t = (int *)malloc((size_t)L * sizeof(int));
    memcpy(gm->wq_t, host->wq_t, (size_t)L * sizeof(int));
    memcpy(gm->wk_t, host->wk_t, (size_t)L * sizeof(int));
    memcpy(gm->wv_t, host->wv_t, (size_t)L * sizeof(int));
    memcpy(gm->wo_t, host->wo_t, (size_t)L * sizeof(int));
    memcpy(gm->gate_t, host->gate_t, (size_t)L * sizeof(int));
    memcpy(gm->up_t, host->up_t, (size_t)L * sizeof(int));
    memcpy(gm->down_t, host->down_t, (size_t)L * sizeof(int));

    gm->norm_att = dev_upload_f32_layers(L, host->norm_att, dim);
    gm->q_norm   = dev_upload_f32_layers(L, host->q_norm, hd);
    gm->k_norm   = dev_upload_f32_layers(L, host->k_norm, hd);
    gm->norm_ffn = dev_upload_f32_layers(L, host->norm_ffn, dim);

    gm->wq   = dev_upload_q1_layers(L, host->wq, dim, dim);
    gm->wk   = dev_upload_q1_layers(L, host->wk, dim, kv_dim);
    gm->wv   = dev_upload_q1_layers(L, host->wv, dim, kv_dim);
    gm->wo   = dev_upload_q1_layers(L, host->wo, dim, dim);
    gm->gate = dev_upload_q1_layers(L, host->gate, dim, hidden);
    gm->up   = dev_upload_q1_layers(L, host->up, dim, hidden);
    gm->down = dev_upload_q1_layers(L, host->down, hidden, dim);

    gm->out_t = host->out_t;
    gm->out = dev_upload(host->out, q1_0_tensor_bytes(dim, vocab));
    gm->norm_out = dev_upload(host->norm_out, (size_t)dim * sizeof(float));

    int max_n = dim > hidden ? dim : hidden;
    gm->q8_nb = max_n / GPU_QK8_0;
    CUDA_CHECK(cudaMalloc(&gm->q8, (size_t)gm->q8_nb * sizeof(GpuBlockQ8_0)));

    CUDA_CHECK(cudaMalloc(&gm->x,      (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->xb,     (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->xb2,    (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->hb,     (size_t)hidden * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->hb2,    (size_t)hidden * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->q,      (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->k,      (size_t)kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->v,      (size_t)kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->logits, (size_t)vocab * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->kc,     (size_t)L * max_seq * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->vc,     (size_t)L * max_seq * kv_dim * sizeof(float)));

    gm->h_rope = (float *)malloc((size_t)n_rot * 2 * sizeof(float));
    CUDA_CHECK(cudaMalloc(&gm->rope_cache, (size_t)n_rot * 2 * sizeof(float)));

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("GPU: weights uploaded (%d layers, dim=%d, vocab=%d)\n", L, dim, vocab);
    return gm;
}

void gpu_model_destroy(GpuModel *gm)
{
    if (!gm) return;
    dev_free(&gm->embd);
    dev_free_layers(&gm->norm_att);
    dev_free_layers(&gm->q_norm);
    dev_free_layers(&gm->k_norm);
    dev_free_layers(&gm->norm_ffn);
    dev_free_layers(&gm->wq);
    dev_free_layers(&gm->wk);
    dev_free_layers(&gm->wv);
    dev_free_layers(&gm->wo);
    dev_free_layers(&gm->gate);
    dev_free_layers(&gm->up);
    dev_free_layers(&gm->down);
    dev_free(&gm->norm_out);
    dev_free(&gm->out);
    free(gm->wq_t); free(gm->wk_t); free(gm->wv_t); free(gm->wo_t);
    free(gm->gate_t); free(gm->up_t); free(gm->down_t);
    cudaFree(gm->x); cudaFree(gm->xb); cudaFree(gm->xb2);
    cudaFree(gm->hb); cudaFree(gm->hb2);
    cudaFree(gm->q); cudaFree(gm->k); cudaFree(gm->v);
    cudaFree(gm->logits);
    cudaFree(gm->kc); cudaFree(gm->vc);
    cudaFree(gm->q8);
    cudaFree(gm->rope_cache);
    free(gm->h_rope);
    free(gm);
}

static void gpu_emb_lookup(GpuModel *gm, int token)
{
    const int dim = gm->cfg.dim;
    int nb = dim / GPU_QK1_0;
    size_t row_stride = (size_t)nb * sizeof(GpuBlockQ1_0);
    const GpuBlockQ1_0 *row = (const GpuBlockQ1_0 *)((const char *)gm->embd.ptr
        + (size_t)token * row_stride);
    emb_q1_0_kernel<<<nb, 128>>>(gm->x, row, nb);
}

void gpu_forward(GpuModel *gm, int token, int pos)
{
    GpuConfig *c = &gm->cfg;
    int dim = c->dim, hd = c->head_dim, kv_dim = c->kv_dim;
    int kv_mul = c->kv_mul, n_heads = c->n_heads, n_kv = c->n_kv_heads;
    int max_seq = c->max_seq, hidden = c->hidden_dim;
    int n_rot = c->n_rot > 0 ? c->n_rot : hd;
    const float scale = 1.0f / sqrtf((float)hd);
    const int npos = pos + 1;

    gpu_emb_lookup(gm, token);

    build_rope_cache_host(c, pos, gm->h_rope);
    CUDA_CHECK(cudaMemcpy(gm->rope_cache, gm->h_rope, (size_t)n_rot * 2 * sizeof(float),
        cudaMemcpyHostToDevice));

    for (int l = 0; l < c->n_layers; l++) {
        rmsnorm_kernel<<<1, 256>>>(gm->xb, gm->x, (float *)gm->norm_att.layer[l], dim, c->norm_eps);

        gpu_mm(gm, gm->q, gm->xb, gm->wq.layer[l], dim, dim, gm->wq_t[l]);
        gpu_mm(gm, gm->k, gm->xb, gm->wk.layer[l], dim, kv_dim, gm->wk_t[l]);
        gpu_mm(gm, gm->v, gm->xb, gm->wv.layer[l], dim, kv_dim, gm->wv_t[l]);

        rmsnorm_head_kernel<<<n_heads, 256>>>(gm->q, (float *)gm->q_norm.layer[l], n_heads, hd, c->norm_eps);
        rmsnorm_head_kernel<<<n_kv, 256>>>(gm->k, (float *)gm->k_norm.layer[l], n_kv, hd, c->norm_eps);

        rope_neox_kernel<<<n_heads, 128>>>(gm->q, gm->rope_cache, n_heads, hd, n_rot);
        rope_neox_kernel<<<n_kv, 128>>>(gm->k, gm->rope_cache, n_kv, hd, n_rot);

        size_t loff = (size_t)l * max_seq * kv_dim;
        float *kc_pos = gm->kc + loff + (size_t)pos * kv_dim;
        float *vc_pos = gm->vc + loff + (size_t)pos * kv_dim;
        CUDA_CHECK(cudaMemcpy(kc_pos, gm->k, (size_t)kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(vc_pos, gm->v, (size_t)kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));

        flash_attn_gqa_kernel<<<n_heads, FA_HD>>>(
            gm->xb, gm->q, gm->kc + loff, gm->vc + loff,
            npos, n_heads, hd, kv_dim, kv_mul, scale);

        gpu_mm(gm, gm->xb2, gm->xb, gm->wo.layer[l], dim, dim, gm->wo_t[l]);
        add_kernel<<<(dim + 255) / 256, 256>>>(gm->x, gm->xb2, dim);

        rmsnorm_kernel<<<1, 256>>>(gm->xb, gm->x, (float *)gm->norm_ffn.layer[l], dim, c->norm_eps);

        gpu_mm(gm, gm->hb,  gm->xb, gm->gate.layer[l], dim, hidden, gm->gate_t[l]);
        gpu_mm(gm, gm->hb2, gm->xb, gm->up.layer[l],   dim, hidden, gm->up_t[l]);

        swiglu_kernel<<<(hidden + 255) / 256, 256>>>(gm->hb, gm->hb2, hidden);

        gpu_mm(gm, gm->xb, gm->hb, gm->down.layer[l], hidden, dim, gm->down_t[l]);
        add_kernel<<<(dim + 255) / 256, 256>>>(gm->x, gm->xb, dim);
    }

    rmsnorm_kernel<<<1, 256>>>(gm->x, gm->x, (float *)gm->norm_out.ptr, dim, c->norm_eps);
    gpu_mm(gm, gm->logits, gm->x, gm->out.ptr, dim, c->vocab_size, gm->out_t);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_copy_logits(GpuModel *gm, float *host_logits)
{
    CUDA_CHECK(cudaMemcpy(host_logits, gm->logits,
        (size_t)gm->cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void gpu_print_device_info(void)
{
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s (compute %d.%d, %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

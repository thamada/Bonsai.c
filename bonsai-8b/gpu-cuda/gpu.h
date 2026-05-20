#ifndef BONSAI_GPU_H
#define BONSAI_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GPU_QK1_0 128
#define GPU_QK8_0 32

#pragma pack(push, 1)
typedef struct {
    uint16_t d;
    uint8_t  qs[GPU_QK1_0 / 8];
} GpuBlockQ1_0;

typedef struct {
    uint16_t d;
    int8_t   qs[GPU_QK8_0];
} GpuBlockQ8_0;
#pragma pack(pop)

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq;
    int head_dim, kv_dim, kv_mul, n_rot, n_ctx_orig_yarn;
    float norm_eps, rope_theta;
    float rope_freq_scale, yarn_ext_factor, yarn_attn_factor;
    float yarn_beta_fast, yarn_beta_slow;
} GpuConfig;

typedef struct {
    const void *embd;    int embd_t;
    float * const *norm_att;
    const void * const *wq;  int *wq_t;
    const void * const *wk;  int *wk_t;
    const void * const *wv;  int *wv_t;
    const void * const *wo;  int *wo_t;
    float * const *q_norm;
    float * const *k_norm;
    float * const *norm_ffn;
    const void * const *gate; int *gate_t;
    const void * const *up;   int *up_t;
    const void * const *down; int *down_t;
    float *norm_out;
    const void *out;     int out_t;
} GpuWeightsHost;

typedef struct GpuModel GpuModel;

GpuModel *gpu_model_create(const GpuConfig *cfg, const GpuWeightsHost *host);
void      gpu_model_destroy(GpuModel *gm);
void      gpu_forward(GpuModel *gm, int token, int pos);
void      gpu_forward_prefill(GpuModel *gm, const int *tokens, int n_tokens);
void      gpu_copy_logits(GpuModel *gm, float *host_logits);
void      gpu_print_device_info(void);

#ifdef __cplusplus
}
#endif

#endif

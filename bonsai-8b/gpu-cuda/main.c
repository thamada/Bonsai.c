#define _POSIX_C_SOURCE 200809L

/*
 * Bonsai 系 dense デコーダ GGUF — NVIDIA GPU CUDA 版（cuBLAS 不要）。
 *
 * 対象: Bonsai-8B-Q1_0.gguf（Q1_0 g128 + F32 norm 等）。テキスト処理のみ。
 * RoPE: llama.cpp の ggml_compute_forward_rope に倣い NeoX 半分ペア配置 + YaRN 系メタを解釈。
 *
 * GPU 方針（cpu-blas 準拠）:
 *   - 重み・KV キャッシュ・活性化は GPU 常駐。起動時に VRAM へアップロード。
 *   - Prefill: 全プロンプトトークンをバッチ並列（gpu_forward_prefill）。
 *   - Decode: 1 トークンずつ gpu_forward。
 *   - 線形層（既定）: Q1_0 重みに対し活性化を Q8_0 化し vec_dot_q1_0_q8_0 CUDA カーネル（gpu_mm）。
 *   - Attention: Flash Attention（online softmax、att 行列非物質化、GQA 対応）。
 *   - サンプリングのみ CPU（logits を D2H コピー）。
 *   - NVFP4 + CUTLASS 経路は gpu-cuda-nvfp4/ を参照。
 *
 * チャット: GGUF tokenizer.chat_template（Qwen3）の user + 空 think ブロック付き assistant 開始。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "gpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define GGUF_MAGIC      0x46554747u
#define MAX_PROMPT_TOKS 8192

enum rope_scaling_kv {
    ROPE_SCL_UNSPEC = 0,
    ROPE_SCL_NONE   = 1,
    ROPE_SCL_YARN   = 2,
    ROPE_SCL_LINEAR = 3
};

enum gguf_vtype {
    GV_U8 = 0, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL,
    GV_STR, GV_ARR, GV_U64, GV_I64, GV_F64
};

/* GGML tensor dtype ids (Bonsai-8B-Q1_0 subset) */
enum ggml_dtype {
    DT_F32  = 0,
    DT_F16  = 1,
    DT_Q1_0 = 41
};


/* ================================================================
 * Model layout (host pointers into mmap)
 * ================================================================ */

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq;
    float rope_theta, norm_eps;
    int head_dim, kv_dim, kv_mul;
    /* RoPE / YaRN メタ（GGUF rope.scaling.* / context_length 等） */
    int n_rot;              /* GGUF の *.rope.dimension_count、無ければ head_dim とみなす */
    int n_ctx_train;        /* *.context_length */
    int n_ctx_orig_yarn;    /* rope.scaling.original_context_length、kV 未定時は後で n_ctx_train を使う */
    float rope_freq_scale;  /* 学習時スケール: 既定 1; GGUF の scaling.factor s に対して 1/s(llama と同様) */
    float yarn_ext_factor;  /* 既定: yarn 無効で 0; YaRN で未指定なら llama と同様 1 */
    float yarn_attn_factor; /* cos/sin へ掛ける mscale(llama ggml と同様、finalize で確定) */
    float rope_attn_factor; /* rope.scaling.attn_factor GGUF、既定 1 */
    float rope_yarn_log_mul; /* rope.scaling.yarn_log_multiplier（ほぼ未使用、f32） */
    float yarn_beta_fast;
    float yarn_beta_slow;
    float rope_scale_kv;    /* GGUF scaling.factor の生値(0=キー無し) */
    int rope_scaling_kind;  /* 0=未指定 1=none 2=yarn 3=linear (llama.cpp の列挙に準拠) */
} Config;

typedef struct {
    char *name;
    int n_dims;
    uint64_t ne[4];
    int type;
    uint64_t offset;
} TensorInfo;

typedef struct {
    char **vocab;
    int *vlen;
    float *scores;
    int size, bos, eos, eot;
    int im_start, im_end;
    int hdr_start, hdr_end;
    int *htab;
    int htab_sz;
    int byte_tok[256];
} Tok;

typedef struct {
    void *embd;    int embd_t;
    float **norm_att;
    void **wq;     int *wq_t;
    void **wk;     int *wk_t;
    void **wv;     int *wv_t;
    void **wo;     int *wo_t;
    float **q_norm;
    float **k_norm;
    float **norm_ffn;
    void **gate;   int *gate_t;
    void **up;     int *up_t;
    void **down;   int *down_t;
    float *norm_out;
    void *out;     int out_t;
} Weights;

typedef struct {
    float *logits; /* host: サンプリング用（GPU から D2H コピー先） */
} State;

typedef struct {
    Config cfg;
    Weights w;
    State s;
    Tok tok;
    GpuModel *gpu;
    int fd;
    uint8_t *fdata;
    size_t fsz;
    TensorInfo *ti;
    int nti;
    uint64_t doff;
} Model;

/* ================================================================
 * GGUF metadata reader
 * ================================================================ */

typedef struct { uint8_t *d; uint64_t p; } Rd;

static uint32_t ru32(Rd *r) { uint32_t v; memcpy(&v, r->d + r->p, 4); r->p += 4; return v; }
static uint64_t ru64(Rd *r) { uint64_t v; memcpy(&v, r->d + r->p, 8); r->p += 8; return v; }
static float    rf32(Rd *r) { float v;    memcpy(&v, r->d + r->p, 4); r->p += 4; return v; }

static char *rstr(Rd *r, int *out_len) {
    uint64_t n = ru64(r);
    if (n > 0x7FFFFFFF) {
        fprintf(stderr, "Error: corrupt string length %llu\n", (unsigned long long)n);
        exit(1);
    }
    char *s = (char *)malloc(n + 1);
    memcpy(s, r->d + r->p, n);
    s[n] = 0;
    r->p += n;
    if (out_len) *out_len = (int)n;
    return s;
}

static void skip(Rd *r, uint32_t t) {
    switch (t) {
    case GV_U8: case GV_I8: case GV_BOOL: r->p++; break;
    case GV_U16: case GV_I16: r->p += 2; break;
    case GV_U32: case GV_I32: case GV_F32: r->p += 4; break;
    case GV_U64: case GV_I64: case GV_F64: r->p += 8; break;
    case GV_STR: { uint64_t n = ru64(r); r->p += n; break; }
    case GV_ARR: {
        uint32_t at = ru32(r); uint64_t al = ru64(r);
        for (uint64_t i = 0; i < al; i++) skip(r, at);
        break;
    }
    }
}

static int64_t read_int_val(Rd *r, uint32_t vt) {
    switch (vt) {
    case GV_U8: case GV_BOOL: { uint8_t v = r->d[r->p]; r->p++; return (int64_t)v; }
    case GV_I8:  { int8_t v; memcpy(&v, r->d + r->p, 1); r->p++; return (int64_t)v; }
    case GV_U16: { uint16_t v; memcpy(&v, r->d + r->p, 2); r->p += 2; return (int64_t)v; }
    case GV_I16: { int16_t v; memcpy(&v, r->d + r->p, 2); r->p += 2; return (int64_t)v; }
    case GV_U32: return (int64_t)ru32(r);
    case GV_I32: { int32_t v; memcpy(&v, r->d + r->p, 4); r->p += 4; return (int64_t)v; }
    case GV_U64: return (int64_t)ru64(r);
    case GV_I64: { int64_t v; memcpy(&v, r->d + r->p, 8); r->p += 8; return v; }
    case GV_F32: return (int64_t)rf32(r);
    case GV_F64: { double v; memcpy(&v, r->d + r->p, 8); r->p += 8; return (int64_t)v; }
    default: skip(r, vt); return 0;
    }
}

static float read_float_val(Rd *r, uint32_t vt) {
    switch (vt) {
    case GV_F32: return rf32(r);
    case GV_F64: { double v; memcpy(&v, r->d + r->p, 8); r->p += 8; return (float)v; }
    case GV_U32: return (float)ru32(r);
    case GV_I32: { int32_t v; memcpy(&v, r->d + r->p, 4); r->p += 4; return (float)v; }
    case GV_U64: return (float)ru64(r);
    case GV_I64: { int64_t v; memcpy(&v, r->d + r->p, 8); r->p += 8; return (float)v; }
    default: skip(r, vt); return 0.0f;
    }
}

/* ================================================================
 * Tokenizer
 * ================================================================ */

static unsigned int hash_bytes(const char *s, int len) {
    unsigned int h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

static void tok_build_hash(Tok *tk) {
    tk->htab_sz = tk->size * 2;
    tk->htab = (int *)malloc(tk->htab_sz * sizeof(int));
    for (int i = 0; i < tk->htab_sz; i++) tk->htab[i] = -1;
    for (int i = 0; i < tk->size; i++) {
        unsigned int h = hash_bytes(tk->vocab[i], tk->vlen[i]) % tk->htab_sz;
        while (tk->htab[h] != -1) h = (h + 1) % tk->htab_sz;
        tk->htab[h] = i;
    }
}

static int tok_lookup(Tok *tk, const char *s, int len) {
    unsigned int h = hash_bytes(s, len) % tk->htab_sz;
    while (tk->htab[h] != -1) {
        int id = tk->htab[h];
        if (tk->vlen[id] == len && memcmp(tk->vocab[id], s, len) == 0) return id;
        h = (h + 1) % tk->htab_sz;
    }
    return -1;
}

static int tok_find_special(Tok *tk, const char *name) {
    int nlen = (int)strlen(name);
    for (int i = tk->size - 1; i >= 0; i--) {
        if (tk->vlen[i] == nlen && memcmp(tk->vocab[i], name, nlen) == 0) return i;
    }
    return -1;
}

static int gpt2_is_printable(int b) {
    return (b >= 0x21 && b <= 0x7E) ||
           (b >= 0xA1 && b <= 0xAC) ||
           (b >= 0xAE && b <= 0xFF);
}

static int gpt2_byte_to_codepoint(int b) {
    if (gpt2_is_printable(b)) return b;
    int n = 0;
    for (int i = 0; i < b; i++)
        if (!gpt2_is_printable(i)) n++;
    return 256 + n;
}

static void build_byte_tokens(Tok *tk) {
    for (int b = 0; b < 256; b++) {
        int cp = gpt2_byte_to_codepoint(b);
        char utf8[4];
        int len;
        if (cp < 0x80) {
            utf8[0] = (char)cp;
            len = 1;
        } else {
            utf8[0] = (char)(0xC0 | (cp >> 6));
            utf8[1] = (char)(0x80 | (cp & 0x3F));
            len = 2;
        }
        tk->byte_tok[b] = tok_lookup(tk, utf8, len);
    }
}

static int gpt2_codepoint_to_byte(int cp) {
    if (gpt2_is_printable(cp)) return cp;
    if (cp >= 256 && cp < 324) {
        int n = cp - 256, count = 0;
        for (int b = 0; b < 256; b++) {
            if (!gpt2_is_printable(b)) {
                if (count == n) return b;
                count++;
            }
        }
    }
    return -1;
}

/* NeoX 半分ペア RoPE + YaRN: ggml-org/llama.cpp ggml-cpu/ops.cpp の forward 相当 */
static float rope_yarn_mscale(float scale, float mscale)
{
    return (scale <= 1.0f) ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
}

static void finalize_rope_hparams(Config *c)
{
    if (c->head_dim == 0 && c->dim > 0 && c->n_heads > 0)
        c->head_dim = c->dim / c->n_heads;
    if (c->n_rot <= 0)
        c->n_rot = c->head_dim;

    float rf = c->rope_scale_kv;
    c->rope_freq_scale = (rf <= 0.0f) ? 1.0f : (1.0f / rf);
    if (c->rope_scaling_kind == ROPE_SCL_NONE)
        c->rope_freq_scale = 1.0f;

    if (c->yarn_ext_factor < 0.0f)
        c->yarn_ext_factor = (c->rope_scaling_kind == ROPE_SCL_YARN) ? 1.0f : 0.0f;

    if (c->n_ctx_orig_yarn <= 0)
        c->n_ctx_orig_yarn = (c->n_ctx_train > 0) ? c->n_ctx_train : 8192;

    float attn = 1.0f;
    /* llama-context.cpp: yarn_ext_factor != 0 のとき attn_factor を確定 */
    if (c->yarn_ext_factor != 0.0f) {
        float stretch = (c->rope_freq_scale > 0.0f) ? (1.0f / c->rope_freq_scale) : 1.0f;
        if (c->rope_yarn_log_mul != 0.0f) {
            float mall = c->rope_yarn_log_mul;
            float mone = 1.0f;
            attn = rope_yarn_mscale(stretch, mone) / rope_yarn_mscale(stretch, mall);
        } else {
            attn = rope_yarn_mscale(stretch, 1.0f);
        }
        attn *= (1.0f / (1.0f + 0.1f * logf(stretch)));
    }
    c->yarn_attn_factor = attn * c->rope_attn_factor;
}

/*
 * dense モデル メタキーは配布 GGUF の規約どおり ASCII プレフィックス + 項目名。
 * llama.cpp 由来の既定プレフィックスでキーを照合する（ソース上はバイト列のみ）。
 */
static const char gguf_dense_meta_key_prefix[] = {
    (char)0x71, (char)0x77, (char)0x65, (char)0x6e, (char)0x33, '.', '\0'
};
#define GGUF_DENSE_META_KEY_PREFIX_LEN 6

static int gguf_key_dense_meta(const char *key, const char *suffix_after_prefix)
{
    return strncmp(key, gguf_dense_meta_key_prefix, GGUF_DENSE_META_KEY_PREFIX_LEN) == 0 &&
           strcmp(key + GGUF_DENSE_META_KEY_PREFIX_LEN, suffix_after_prefix) == 0;
}

static void parse_gguf(Model *m, char ***out_merges, int *out_n_merges) {
    Rd r = { m->fdata, 0 };

    uint32_t magic = ru32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Error: invalid GGUF magic 0x%08x\n", magic);
        exit(1);
    }
    uint32_t ver = ru32(&r);
    if (ver < 2) {
        fprintf(stderr, "Error: GGUF v%u not supported (need v2+)\n", ver);
        exit(1);
    }
    uint64_t n_tensors = ru64(&r);
    uint64_t n_kv      = ru64(&r);

    printf("GGUF v%u | %llu tensors | %llu metadata entries\n",
           ver, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    Config *c = &m->cfg;
    Tok *tk = &m->tok;

    c->rope_theta = 5000000.0f;
    c->norm_eps   = 1e-6f;
    c->max_seq    = 512;
    tk->bos = 151643;
    tk->eos = 151645;
    tk->eot = 151645;
    tk->im_start = -1;
    tk->im_end = -1;
    tk->hdr_start = -1;
    tk->hdr_end = -1;
    c->head_dim = 0;
    c->n_rot = 0;
    c->n_ctx_train = 0;
    c->n_ctx_orig_yarn = 0;
    c->rope_freq_scale = 1.0f;
    c->yarn_ext_factor = -1.0f; /* sentinel: finalize で yarn 可否に応じて設定 */
    c->yarn_attn_factor = 1.0f;
    c->rope_attn_factor = 1.0f;
    c->rope_yarn_log_mul = 0.0f;
    c->yarn_beta_fast = 32.0f;
    c->yarn_beta_slow = 1.0f;
    c->rope_scale_kv = 0.0f;
    c->rope_scaling_kind = ROPE_SCL_UNSPEC;

    *out_merges = NULL;
    *out_n_merges = 0;

    uint32_t alignment = 32;

    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = rstr(&r, NULL);
        uint32_t vt = ru32(&r);

        if      (!strcmp(key, "general.alignment"))                             { alignment     = (uint32_t)read_int_val(&r, vt); }
        else if (gguf_key_dense_meta(key, "embedding_length"))                 { c->dim        = (int)read_int_val(&r, vt); }
        else if (gguf_key_dense_meta(key, "feed_forward_length"))             { c->hidden_dim = (int)read_int_val(&r, vt); }
        else if (gguf_key_dense_meta(key, "block_count"))                      { c->n_layers   = (int)read_int_val(&r, vt); }
        else if (gguf_key_dense_meta(key, "attention.head_count"))            { c->n_heads    = (int)read_int_val(&r, vt); }
        else if (gguf_key_dense_meta(key, "attention.head_count_kv"))         { c->n_kv_heads = (int)read_int_val(&r, vt); }
        else if (gguf_key_dense_meta(key, "attention.key_length"))            { c->head_dim   = (int)read_int_val(&r, vt); }
        else if (gguf_key_dense_meta(key, "attention.layer_norm_rms_epsilon")) { c->norm_eps   = read_float_val(&r, vt); }
        else if (gguf_key_dense_meta(key, "rope.freq_base"))                   { c->rope_theta = read_float_val(&r, vt); }
        else if (gguf_key_dense_meta(key, "context_length"))
            c->n_ctx_train = (int)read_int_val(&r, vt);
        else if (gguf_key_dense_meta(key, "rope.dimension_count"))
            c->n_rot = (int)read_int_val(&r, vt);
        else if (gguf_key_dense_meta(key, "rope.scaling.type")) {
            if (vt != GV_STR) { skip(&r, vt); free(key); continue; }
            char *rs = rstr(&r, NULL);
            if      (!strcmp(rs, "none"))   c->rope_scaling_kind = ROPE_SCL_NONE;
            else if (!strcmp(rs, "yarn"))   c->rope_scaling_kind = ROPE_SCL_YARN;
            else if (!strcmp(rs, "linear")) c->rope_scaling_kind = ROPE_SCL_LINEAR;
            free(rs);
            free(key); continue;
        }
        else if (gguf_key_dense_meta(key, "rope.scaling.factor")) {
            float f = read_float_val(&r, vt);
            if (f > 0.0f) c->rope_scale_kv = f;
        }
        else if (gguf_key_dense_meta(key, "rope.scaling.original_context_length"))
            c->n_ctx_orig_yarn = (int)read_int_val(&r, vt);
        else if (gguf_key_dense_meta(key, "rope.scaling.attn_factor"))
            c->rope_attn_factor = read_float_val(&r, vt);
        else if (gguf_key_dense_meta(key, "rope.scaling.yarn_ext_factor"))
            c->yarn_ext_factor = read_float_val(&r, vt);
        else if (gguf_key_dense_meta(key, "rope.scaling.yarn_log_multiplier"))
            c->rope_yarn_log_mul = read_float_val(&r, vt);
        else if (gguf_key_dense_meta(key, "rope.scaling.yarn_beta_fast"))
            c->yarn_beta_fast = read_float_val(&r, vt);
        else if (gguf_key_dense_meta(key, "rope.scaling.yarn_beta_slow"))
            c->yarn_beta_slow = read_float_val(&r, vt);
        else if (!strcmp(key, "tokenizer.ggml.bos_token_id"))                   { tk->bos       = (int)read_int_val(&r, vt); }
        else if (!strcmp(key, "tokenizer.ggml.eos_token_id"))                  { tk->eos       = (int)read_int_val(&r, vt); tk->eot = tk->eos; }
        else if (!strcmp(key, "tokenizer.ggml.tokens")) {
            if (vt != GV_ARR) { skip(&r, vt); free(key); continue; }
            ru32(&r);
            uint64_t n = ru64(&r);
            tk->size = (int)n;
            c->vocab_size = (int)n;
            tk->vocab = (char **)calloc(n, sizeof(char *));
            tk->vlen  = (int *)calloc(n, sizeof(int));
            for (uint64_t j = 0; j < n; j++)
                tk->vocab[j] = rstr(&r, &tk->vlen[j]);
        }
        else if (!strcmp(key, "tokenizer.ggml.scores")) {
            if (vt != GV_ARR) { skip(&r, vt); free(key); continue; }
            ru32(&r);
            uint64_t n = ru64(&r);
            tk->scores = (float *)malloc(n * sizeof(float));
            for (uint64_t j = 0; j < n; j++) tk->scores[j] = rf32(&r);
        }
        else if (!strcmp(key, "tokenizer.ggml.merges")) {
            if (vt != GV_ARR) { skip(&r, vt); free(key); continue; }
            ru32(&r);
            uint64_t n = ru64(&r);
            *out_n_merges = (int)n;
            *out_merges = (char **)malloc(n * sizeof(char *));
            for (uint64_t j = 0; j < n; j++)
                (*out_merges)[j] = rstr(&r, NULL);
        }
        else { skip(&r, vt); }

        free(key);
    }

    if (c->head_dim == 0)
        c->head_dim = c->dim / c->n_heads;
    finalize_rope_hparams(c);
    c->kv_dim = c->n_kv_heads * c->head_dim;
    c->kv_mul = c->n_heads / c->n_kv_heads;

    m->nti = (int)n_tensors;
    m->ti  = (TensorInfo *)calloc(n_tensors, sizeof(TensorInfo));
    for (uint64_t i = 0; i < n_tensors; i++) {
        m->ti[i].name   = rstr(&r, NULL);
        m->ti[i].n_dims = (int)ru32(&r);
        for (int d = 0; d < m->ti[i].n_dims; d++)
            m->ti[i].ne[d] = ru64(&r);
        m->ti[i].type   = (int)ru32(&r);
        m->ti[i].offset = ru64(&r);
    }

    m->doff = (r.p + (alignment - 1)) & ~(uint64_t)(alignment - 1);
}

static void init_tokenizer(Tok *tk, char **merges, int n_merges) {
    if (!tk->scores && tk->size > 0)
        tk->scores = (float *)calloc(tk->size, sizeof(float));
    tok_build_hash(tk);
    build_byte_tokens(tk);

    if (n_merges > 0 && merges) {
        for (int i = 0; i < n_merges; i++) {
            char *sp = strchr(merges[i], ' ');
            if (!sp) { free(merges[i]); continue; }
            int la = (int)(sp - merges[i]);
            int lb = (int)strlen(sp + 1);
            char *buf = (char *)malloc(la + lb + 1);
            memcpy(buf, merges[i], la);
            memcpy(buf + la, sp + 1, lb);
            buf[la + lb] = 0;
            int id = tok_lookup(tk, buf, la + lb);
            if (id >= 0)
                tk->scores[id] = (float)(n_merges - i);
            free(buf);
            free(merges[i]);
        }
        free(merges);
    }

    tk->im_start = tok_find_special(tk, "<|im_start|>");
    tk->im_end   = tok_find_special(tk, "<|im_end|>");
    tk->hdr_start = tk->im_start;
    tk->hdr_end   = tk->im_end;
    if (tk->eot < 0 || tk->eot >= tk->size)
        tk->eot = (tk->im_end >= 0) ? tk->im_end : tk->eos;

    printf("Tokenizer: %d tokens | BOS=%d EOS=%d EOT=%d im_start=%d im_end=%d\n",
           tk->size, tk->bos, tk->eos, tk->eot, tk->im_start, tk->im_end);
}

static int *bpe_encode(Tok *tk, const char *text, int text_len, int *out_n) {
    int *tokens = (int *)malloc((text_len + 1) * sizeof(int));
    int n = 0;

    for (int i = 0; i < text_len; i++) {
        int id = tk->byte_tok[(unsigned char)text[i]];
        if (id >= 0) tokens[n++] = id;
        else fprintf(stderr, "Warning: byte 0x%02x not in vocabulary\n", (unsigned char)text[i]);
    }

    char buf[MAX_PROMPT_TOKS];
    while (n > 1) {
        float best_score = -1e20f;
        int best_id = -1, best_idx = -1;

        for (int i = 0; i < n - 1; i++) {
            int total = tk->vlen[tokens[i]] + tk->vlen[tokens[i + 1]];
            if (total >= (int)sizeof(buf)) continue;
            memcpy(buf, tk->vocab[tokens[i]], tk->vlen[tokens[i]]);
            memcpy(buf + tk->vlen[tokens[i]], tk->vocab[tokens[i + 1]], tk->vlen[tokens[i + 1]]);
            int id = tok_lookup(tk, buf, total);
            if (id >= 0 && tk->scores[id] > best_score) {
                best_score = tk->scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx < 0) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < n - 1; i++) tokens[i] = tokens[i + 1];
        n--;
    }

    *out_n = n;
    return tokens;
}

static void append_bpe(Tok *tk, int *out, int *n, const char *text) {
    int nt;
    int *t = bpe_encode(tk, text, (int)strlen(text), &nt);
    for (int i = 0; i < nt; i++) out[(*n)++] = t[i];
    free(t);
}

/*
 * GGUF tokenizer.chat_template（Qwen3 / Bonsai）の単一 user ターン + 生成プレフィックスに合わせる。
 * 既定 system 文は挿入しない。assistant 開始は空の think ブロック付き
 *（add_generation_prompt 相当。PrismML llama.cpp -cnv と同じ）。
 */
static int *chat_encode(Tok *tk, const char *prompt, int *out_n) {
    int *toks = (int *)malloc(MAX_PROMPT_TOKS * sizeof(int));
    int n = 0;

    if (tk->im_start >= 0) toks[n++] = tk->im_start;
    append_bpe(tk, toks, &n, "user\n");
    append_bpe(tk, toks, &n, prompt);
    if (tk->im_end >= 0) toks[n++] = tk->im_end;
    append_bpe(tk, toks, &n, "\n");

    if (tk->im_start >= 0) toks[n++] = tk->im_start;
    /* GGUF chat_template add_generation_prompt と同じリテラル */
    append_bpe(tk, toks, &n,
        "assistant\n"
        "\x3c\x7cthink\x7c\x3e\n\n"
        "\x3c\x2fthink\x7c\x3e\n\n");

    *out_n = n;
    return toks;
}

static void expect_q1_0(int type, const char *name) {
    if (type != DT_Q1_0) {
        fprintf(stderr, "Error: %s must be Q1_0 (got GGML type %d)\n", name, type);
        exit(1);
    }
}

static void *find_tensor(Model *m, const char *name, int *out_type) {
    for (int i = 0; i < m->nti; i++) {
        if (strcmp(m->ti[i].name, name) == 0) {
            if (out_type) *out_type = m->ti[i].type;
            return m->fdata + m->doff + m->ti[i].offset;
        }
    }
    return NULL;
}

static void load_weights(Model *m) {
    Config *c = &m->cfg;
    Weights *w = &m->w;
    int L = c->n_layers;

    w->embd = find_tensor(m, "token_embd.weight", &w->embd_t);

    w->norm_att = (float **)calloc(L, sizeof(float *));
    w->q_norm   = (float **)calloc(L, sizeof(float *));
    w->k_norm   = (float **)calloc(L, sizeof(float *));
    w->wq       = (void **)calloc(L, sizeof(void *));  w->wq_t   = (int *)calloc(L, sizeof(int));
    w->wk       = (void **)calloc(L, sizeof(void *));  w->wk_t   = (int *)calloc(L, sizeof(int));
    w->wv       = (void **)calloc(L, sizeof(void *));  w->wv_t   = (int *)calloc(L, sizeof(int));
    w->wo       = (void **)calloc(L, sizeof(void *));  w->wo_t   = (int *)calloc(L, sizeof(int));
    w->norm_ffn = (float **)calloc(L, sizeof(float *));
    w->gate     = (void **)calloc(L, sizeof(void *));  w->gate_t  = (int *)calloc(L, sizeof(int));
    w->up       = (void **)calloc(L, sizeof(void *));  w->up_t    = (int *)calloc(L, sizeof(int));
    w->down     = (void **)calloc(L, sizeof(void *));  w->down_t  = (int *)calloc(L, sizeof(int));

    char name[128];
    for (int l = 0; l < L; l++) {
        sprintf(name, "blk.%d.attn_norm.weight", l);
        w->norm_att[l] = (float *)find_tensor(m, name, NULL);
        sprintf(name, "blk.%d.attn_q_norm.weight", l);
        w->q_norm[l] = (float *)find_tensor(m, name, NULL);
        sprintf(name, "blk.%d.attn_k_norm.weight", l);
        w->k_norm[l] = (float *)find_tensor(m, name, NULL);
        sprintf(name, "blk.%d.attn_q.weight", l);
        w->wq[l] = find_tensor(m, name, &w->wq_t[l]);
        sprintf(name, "blk.%d.attn_k.weight", l);
        w->wk[l] = find_tensor(m, name, &w->wk_t[l]);
        sprintf(name, "blk.%d.attn_v.weight", l);
        w->wv[l] = find_tensor(m, name, &w->wv_t[l]);
        sprintf(name, "blk.%d.attn_output.weight", l);
        w->wo[l] = find_tensor(m, name, &w->wo_t[l]);
        sprintf(name, "blk.%d.ffn_norm.weight", l);
        w->norm_ffn[l] = (float *)find_tensor(m, name, NULL);
        sprintf(name, "blk.%d.ffn_gate.weight", l);
        w->gate[l] = find_tensor(m, name, &w->gate_t[l]);
        sprintf(name, "blk.%d.ffn_up.weight", l);
        w->up[l] = find_tensor(m, name, &w->up_t[l]);
        sprintf(name, "blk.%d.ffn_down.weight", l);
        w->down[l] = find_tensor(m, name, &w->down_t[l]);
    }

    w->norm_out = (float *)find_tensor(m, "output_norm.weight", NULL);
    w->out = find_tensor(m, "output.weight", &w->out_t);
    if (!w->out) {
        fprintf(stderr, "Error: output.weight missing (untied LM head expected)\n");
        exit(1);
    }

    if (!w->embd || !w->norm_out) {
        fprintf(stderr, "Error: missing critical tensors\n");
        exit(1);
    }
    for (int l = 0; l < L; l++) {
        if (!w->norm_att[l] || !w->q_norm[l] || !w->k_norm[l] || !w->wq[l] || !w->wk[l] || !w->wv[l] ||
            !w->wo[l] || !w->norm_ffn[l] || !w->gate[l] || !w->up[l] || !w->down[l]) {
            fprintf(stderr, "Error: missing tensor(s) in layer %d\n", l);
            exit(1);
        }
    }

    expect_q1_0(w->embd_t, "token_embd.weight");
    expect_q1_0(w->out_t, "output.weight");
    for (int l = 0; l < L; l++) {
        sprintf(name, "blk.%d.attn_q.weight", l);
        expect_q1_0(w->wq_t[l], name);
        sprintf(name, "blk.%d.attn_k.weight", l);
        expect_q1_0(w->wk_t[l], name);
        sprintf(name, "blk.%d.attn_v.weight", l);
        expect_q1_0(w->wv_t[l], name);
        sprintf(name, "blk.%d.attn_output.weight", l);
        expect_q1_0(w->wo_t[l], name);
        sprintf(name, "blk.%d.ffn_gate.weight", l);
        expect_q1_0(w->gate_t[l], name);
        sprintf(name, "blk.%d.ffn_up.weight", l);
        expect_q1_0(w->up_t[l], name);
        sprintf(name, "blk.%d.ffn_down.weight", l);
        expect_q1_0(w->down_t[l], name);
    }
}

static void free_weight_ptrs(Weights *w, int L) {
    free(w->norm_att); free(w->q_norm); free(w->k_norm);
    free(w->wq); free(w->wq_t); free(w->wk); free(w->wk_t);
    free(w->wv); free(w->wv_t); free(w->wo); free(w->wo_t);
    free(w->norm_ffn);
    free(w->gate); free(w->gate_t); free(w->up); free(w->up_t); free(w->down); free(w->down_t);
    memset(w, 0, sizeof(*w));
    (void)L;
}

static void alloc_state(State *s, Config *c) {
    s->logits = (float *)calloc((size_t)c->vocab_size, sizeof(float));
}

static void free_state(State *s) {
    free(s->logits);
}

static GpuConfig gpu_config_from(const Config *c) {
    GpuConfig g;
    g.dim = c->dim; g.hidden_dim = c->hidden_dim;
    g.n_layers = c->n_layers; g.n_heads = c->n_heads;
    g.n_kv_heads = c->n_kv_heads; g.vocab_size = c->vocab_size;
    g.max_seq = c->max_seq; g.head_dim = c->head_dim;
    g.kv_dim = c->kv_dim; g.kv_mul = c->kv_mul;
    g.n_rot = c->n_rot; g.n_ctx_orig_yarn = c->n_ctx_orig_yarn;
    g.norm_eps = c->norm_eps; g.rope_theta = c->rope_theta;
    g.rope_freq_scale = c->rope_freq_scale;
    g.yarn_ext_factor = c->yarn_ext_factor;
    g.yarn_attn_factor = c->yarn_attn_factor;
    g.yarn_beta_fast = c->yarn_beta_fast;
    g.yarn_beta_slow = c->yarn_beta_slow;
    return g;
}

static GpuWeightsHost gpu_weights_from(const Weights *w) {
    GpuWeightsHost gh;
    gh.embd = w->embd; gh.embd_t = w->embd_t;
    gh.norm_att = w->norm_att;
    gh.wq = (const void * const *)w->wq; gh.wq_t = w->wq_t;
    gh.wk = (const void * const *)w->wk; gh.wk_t = w->wk_t;
    gh.wv = (const void * const *)w->wv; gh.wv_t = w->wv_t;
    gh.wo = (const void * const *)w->wo; gh.wo_t = w->wo_t;
    gh.q_norm = w->q_norm; gh.k_norm = w->k_norm;
    gh.norm_ffn = w->norm_ffn;
    gh.gate = (const void * const *)w->gate; gh.gate_t = w->gate_t;
    gh.up = (const void * const *)w->up; gh.up_t = w->up_t;
    gh.down = (const void * const *)w->down; gh.down_t = w->down_t;
    gh.norm_out = w->norm_out;
    gh.out = w->out; gh.out_t = w->out_t;
    return gh;
}

static void forward(Model *m, int token, int pos) {
    gpu_forward(m->gpu, token, pos);
    gpu_copy_logits(m->gpu, m->s.logits);
}

static void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

static float rng_f32(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (float)((*state * 0x2545F4914F6CDD1DULL) >> 40) / (float)(1 << 24);
}

typedef struct { float p; int idx; } ProbIdx;

static int cmp_prob_desc(const void *a, const void *b) {
    float pa = ((const ProbIdx *)a)->p;
    float pb = ((const ProbIdx *)b)->p;
    return (pa < pb) - (pa > pb);
}

static int sample_token(float *logits, int n, float temp, float topp, uint64_t *rng) {
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    for (int i = 0; i < n; i++) logits[i] /= temp;
    softmax(logits, n);

    float coin = rng_f32(rng);

    if (topp <= 0.0f || topp >= 1.0f) {
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += logits[i];
            if (coin < cdf) return i;
        }
        return n - 1;
    }

    ProbIdx *pi = (ProbIdx *)malloc(n * sizeof(ProbIdx));
    int np = 0;
    float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (logits[i] >= cutoff) {
            pi[np].p = logits[i];
            pi[np].idx = i;
            np++;
        }
    }
    qsort(pi, np, sizeof(ProbIdx), cmp_prob_desc);

    float cum = 0.0f;
    int last = np - 1;
    for (int i = 0; i < np; i++) {
        cum += pi[i].p;
        if (cum > topp) { last = i; break; }
    }

    float r = rng_f32(rng) * cum;
    float cdf = 0.0f;
    int result = pi[last].idx;
    for (int i = 0; i <= last; i++) {
        cdf += pi[i].p;
        if (r < cdf) { result = pi[i].idx; break; }
    }
    free(pi);
    return result;
}

static int is_special(Tok *tk, int id) {
    return id == tk->bos || id == tk->eos || id == tk->eot ||
           id == tk->im_start || id == tk->im_end ||
           id == tk->hdr_start || id == tk->hdr_end;
}

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} GenBuf;

static void genbuf_init(GenBuf *gb) {
    gb->data = NULL;
    gb->len = 0;
    gb->cap = 0;
}

static void genbuf_free(GenBuf *gb) {
    free(gb->data);
    gb->data = NULL;
    gb->len = gb->cap = 0;
}

static int genbuf_reserve(GenBuf *gb, size_t need) {
    if (need <= gb->cap) return 1;
    size_t ncap = gb->cap ? gb->cap : 256;
    while (ncap < need) ncap *= 2;
    char *p = (char *)realloc(gb->data, ncap);
    if (!p) return 0;
    gb->data = p;
    gb->cap = ncap;
    return 1;
}

static int genbuf_putc(GenBuf *gb, int ch) {
    if (!genbuf_reserve(gb, gb->len + 2)) return 0;
    gb->data[gb->len++] = (char)ch;
    gb->data[gb->len] = '\0';
    return 1;
}

static int genbuf_write(GenBuf *gb, const void *src, size_t n) {
    if (!genbuf_reserve(gb, gb->len + n + 1)) return 0;
    memcpy(gb->data + gb->len, src, n);
    gb->len += n;
    gb->data[gb->len] = '\0';
    return 1;
}

static void append_tok(GenBuf *gb, Tok *tk, int id) {
    if (id < 0 || id >= tk->size || is_special(tk, id)) return;
    const uint8_t *s = (const uint8_t *)tk->vocab[id];
    int len = tk->vlen[id];
    for (int i = 0; i < len; ) {
        int cp, adv;
        if (s[i] < 0x80) { cp = s[i]; adv = 1; }
        else if ((s[i] & 0xE0) == 0xC0 && i + 1 < len) {
            cp = ((s[i] & 0x1F) << 6) | (s[i+1] & 0x3F); adv = 2;
        } else if ((s[i] & 0xF0) == 0xE0 && i + 2 < len) {
            cp = ((s[i] & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F); adv = 3;
        } else { cp = s[i]; adv = 1; }
        int raw = gpt2_codepoint_to_byte(cp);
        if (raw >= 0) {
            if (!genbuf_putc(gb, raw)) return;
        } else if (!genbuf_write(gb, s + i, (size_t)adv)) {
            return;
        }
        i += adv;
    }
}

static void print_tok(Tok *tk, int id) {
    if (id < 0 || id >= tk->size || is_special(tk, id)) return;
    const uint8_t *s = (const uint8_t *)tk->vocab[id];
    int len = tk->vlen[id];
    for (int i = 0; i < len; ) {
        int cp, adv;
        if (s[i] < 0x80) { cp = s[i]; adv = 1; }
        else if ((s[i] & 0xE0) == 0xC0 && i + 1 < len) {
            cp = ((s[i] & 0x1F) << 6) | (s[i+1] & 0x3F); adv = 2;
        } else if ((s[i] & 0xF0) == 0xE0 && i + 2 < len) {
            cp = ((s[i] & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F); adv = 3;
        } else { cp = s[i]; adv = 1; }
        int raw = gpt2_codepoint_to_byte(cp);
        if (raw >= 0)
            putchar(raw);
        else
            fwrite(s + i, 1, adv, stdout);
        i += adv;
    }
    fflush(stdout);
}

static void print_tok_and_append(Tok *tk, int id, GenBuf *gb) {
    print_tok(tk, id);
    append_tok(gb, tk, id);
}

#define PREFILL_BAR_WIDTH 40

static void prefill_progress_update(int done, int total) {
    if (total <= 0) return;
    if (done > total) done = total;
    int filled = (done * PREFILL_BAR_WIDTH) / total;
    int pct = (done * 100) / total;
    fprintf(stderr, "\rPrefill [");
    for (int i = 0; i < PREFILL_BAR_WIDTH; i++)
        fputc(i < filled ? '=' : ' ', stderr);
    fprintf(stderr, "] %3d%% (%d/%d)", pct, done, total);
    fflush(stderr);
}

static void prefill_progress_done(int n_tokens, double elapsed_sec) {
    double tps = (elapsed_sec > 0.0) ? (double)n_tokens / elapsed_sec : 0.0;
    fprintf(stderr, "\rPrefill [");
    for (int i = 0; i < PREFILL_BAR_WIDTH; i++)
        fputc('=', stderr);
    fprintf(stderr, "] 100%% (%d/%d)\n", n_tokens, n_tokens);
    fprintf(stderr, "Prefill complete: %d tokens in %.2fs (%.2f tok/s)\n",
            n_tokens, elapsed_sec, tps);
}

static void decode_progress_done(int n_tokens, double elapsed_sec) {
    double tps = (elapsed_sec > 0.0) ? (double)n_tokens / elapsed_sec : 0.0;
    fflush(stdout);
    fputc('\n', stdout);
    fflush(stdout);
    fprintf(stderr, "\nDecode complete: %d tokens in %.2fs (%.2f tok/s)\n",
            n_tokens, elapsed_sec, tps);
}

#define BENCH_LOG_DEFAULT "/tmp/benchmark.log"

typedef struct {
    const char *model_path;
    const char *prompt_text;
    const char *gpu_desc;
    int   max_new;
    float temp;
    float topp;
    uint64_t seed;
    int   max_seq;
    int   n_prefill;
    int   n_decode;
    double prefill_sec;
    double decode_sec;
    double total_sec;
    double prefill_tps;
    double decode_tps;
    double total_tps;
    const char *output;
    GpuVramProfile vram;
} BenchLogInfo;

static void write_vram_bytes_line(FILE *f, const char *key, size_t bytes)
{
    fprintf(f, "%s=%zu\n", key, bytes);
    fprintf(f, "%s_mib=%.2f\n", key, (double)bytes / (1024.0 * 1024.0));
}

static void write_benchmark_log(const BenchLogInfo *info) {
    const char *path = getenv("BENCH_LOG_FILE");
    if (!path || !path[0]) path = BENCH_LOG_DEFAULT;

    time_t now = time(NULL);
    struct tm tm_local;
    localtime_r(&now, &tm_local);
    char timestamp[32];
    strftime(timestamp, sizeof timestamp, "%Y-%m-%dT%H:%M:%S", &tm_local);

    char hostname[256];
    hostname[0] = '\0';
    if (gethostname(hostname, sizeof hostname) != 0)
        snprintf(hostname, sizeof hostname, "unknown");

    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "warning: could not write benchmark log: %s\n", path);
        return;
    }

    fprintf(f, "# bonsai-gpu-cuda benchmark log\n");
    fprintf(f, "timestamp=%s\n", timestamp);
    fprintf(f, "hostname=%s\n", hostname);
    fprintf(f, "model=%s\n", info->model_path ? info->model_path : "");
    fprintf(f, "max_new=%d\n", info->max_new);
    fprintf(f, "temperature=%.6g\n", (double)info->temp);
    fprintf(f, "top_p=%.6g\n", (double)info->topp);
    fprintf(f, "seed=%llu\n", (unsigned long long)info->seed);
    fprintf(f, "max_seq=%d\n", info->max_seq);
    fprintf(f, "gpu=%s\n", info->gpu_desc ? info->gpu_desc : "");
    fprintf(f, "\n");
    fprintf(f, "prompt_tokens=%d\n", info->n_prefill);
    fprintf(f, "gen_tokens=%d\n", info->n_decode);
    fprintf(f, "prefill_sec=%.4f\n", info->prefill_sec);
    fprintf(f, "decode_sec=%.4f\n", info->decode_sec);
    fprintf(f, "total_sec=%.4f\n", info->total_sec);
    fprintf(f, "prefill_tps=%.2f\n", info->prefill_tps);
    fprintf(f, "decode_tps=%.2f\n", info->decode_tps);
    fprintf(f, "total_tps=%.2f\n", info->total_tps);
    fprintf(f, "\n");
    write_vram_bytes_line(f, "vram_total", info->vram.total_bytes);
    if (info->vram.device_total_bytes > 0) {
        write_vram_bytes_line(f, "vram_device_used", info->vram.device_used_bytes);
        write_vram_bytes_line(f, "vram_device_total", info->vram.device_total_bytes);
    }
    fprintf(f, "\n");
    fprintf(f, "[vram_breakdown]\n");
    write_vram_bytes_line(f, "vram_weights_q1_embd", info->vram.weights_q1_embd_bytes);
    write_vram_bytes_line(f, "vram_weights_f32_norm", info->vram.weights_f32_norm_bytes);
    write_vram_bytes_line(f, "vram_weights_q1_linear", info->vram.weights_q1_linear_bytes);
    write_vram_bytes_line(f, "vram_kv_cache", info->vram.kv_cache_bytes);
    write_vram_bytes_line(f, "vram_decode_activations", info->vram.decode_activations_bytes);
    write_vram_bytes_line(f, "vram_prefill_batch", info->vram.prefill_batch_bytes);
    fprintf(f, "\n");
    fprintf(f, "--- prompt ---\n");
    fprintf(f, "%s\n", info->prompt_text ? info->prompt_text : "");
    fprintf(f, "--- end prompt ---\n");
    fprintf(f, "\n");
    fprintf(f, "--- output ---\n");
    fprintf(f, "%s\n", info->output ? info->output : "");
    fprintf(f, "--- end output ---\n");
    fclose(f);
}

static void throughput_summary(const BenchLogInfo *meta, GpuModel *gpu,
                             int n_prefill, double prefill_sec,
                             int n_decode, double decode_sec,
                             double total_sec,
                             const char *output) {
    double prefill_tps = (prefill_sec > 0.0) ? (double)n_prefill / prefill_sec : 0.0;
    double decode_tps  = (decode_sec > 0.0)  ? (double)n_decode / decode_sec  : 0.0;
    int n_total = n_prefill + n_decode;
    double total_tps = (total_sec > 0.0) ? (double)n_total / total_sec : 0.0;
    fprintf(stderr, "--- throughput ---\n");
    fprintf(stderr, "  prefill: %.2f tok/s\n", prefill_tps);
    fprintf(stderr, "  decode:  %.2f tok/s\n", decode_tps);
    fprintf(stderr, "  total:   %.2f tok/s\n", total_tps);

    BenchLogInfo info = *meta;
    info.n_prefill = n_prefill;
    info.n_decode = n_decode;
    info.prefill_sec = prefill_sec;
    info.decode_sec = decode_sec;
    info.total_sec = total_sec;
    info.prefill_tps = prefill_tps;
    info.decode_tps = decode_tps;
    info.total_tps = total_tps;
    info.output = output ? output : "";
    if (gpu)
        gpu_model_vram_profile(gpu, &info.vram);
    write_benchmark_log(&info);
}

static void generate(Model *m, int *prompt, int n_prompt,
                     int max_new, float temp, float topp, uint64_t seed,
                     const BenchLogInfo *meta) {
    uint64_t rng = seed ? seed : 1;
    int gen = 0;
    double prefill_sec = 0.0;
    GenBuf gen_out;
    genbuf_init(&gen_out);

    struct timespec t0, t1, t_prefill, t_decode;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (n_prompt > 0) {
        if (n_prompt > m->cfg.max_seq) {
            fprintf(stderr, "\n[prompt length %d exceeds max_seq %d]\n",
                n_prompt, m->cfg.max_seq);
            genbuf_free(&gen_out);
            return;
        }
        clock_gettime(CLOCK_MONOTONIC, &t_prefill);
        prefill_progress_update(0, n_prompt);
        if (n_prompt > 1) {
            gpu_forward_prefill(m->gpu, prompt, n_prompt);
        } else {
            forward(m, prompt[0], 0);
        }
        gpu_copy_logits(m->gpu, m->s.logits);
        struct timespec t_now;
        clock_gettime(CLOCK_MONOTONIC, &t_now);
        prefill_sec = (t_now.tv_sec - t_prefill.tv_sec)
            + (t_now.tv_nsec - t_prefill.tv_nsec) / 1e9;
        prefill_progress_done(n_prompt, prefill_sec);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_decode);

    for (int gen_i = 0; gen_i < max_new; gen_i++) {
        int pos = n_prompt - 1 + gen_i;
        if (pos >= m->cfg.max_seq) {
            fprintf(stderr, "\n[max sequence length %d reached]\n", m->cfg.max_seq);
            break;
        }

        int next = sample_token(m->s.logits, m->cfg.vocab_size, temp, topp, &rng);
        if (next == m->tok.eos || next == m->tok.eot) break;
        gen++;
        print_tok_and_append(&m->tok, next, &gen_out);

        pos = n_prompt + gen_i;
        if (pos >= m->cfg.max_seq) break;
        forward(m, next, pos);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double decode_sec = (t1.tv_sec - t_decode.tv_sec)
        + (t1.tv_nsec - t_decode.tv_nsec) / 1e9;
    if (gen > 0)
        decode_progress_done(gen, decode_sec);

    printf("\n\n--- %d prompt tokens + %d generated tokens ---\n", n_prompt, gen);
    printf("--- %.1fs total ---\n", elapsed);
    if (n_prompt > 0 || gen > 0)
        throughput_summary(meta, m->gpu, n_prompt, prefill_sec, gen, decode_sec, elapsed,
                           gen_out.data ? gen_out.data : "");
    genbuf_free(&gen_out);
}

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  -p <prompt>   User prompt (default: Hello)\n");
        fprintf(stderr, "  -n <tokens>   Max tokens to generate (default: 256)\n");
        fprintf(stderr, "  -t <temp>     Temperature (default: 0.6)\n");
        fprintf(stderr, "  -k <topp>     Top-p sampling (default: 0.9)\n");
        fprintf(stderr, "  -s <seed>     Random seed (default: time)\n");
        fprintf(stderr, "  -l <len>      Max sequence length (default: 512)\n");
        return 1;
    }

    char *model_path = argv[1];
    const char *prompt = "Hello";
    int   max_tokens = 256;
    float temp       = 0.6f;
    float topp       = 0.9f;
    uint64_t seed    = (uint64_t)time(NULL);
    int   max_seq    = 512;

    for (int i = 2; i + 1 < argc; i += 2) {
        if      (!strcmp(argv[i], "-p")) prompt     = argv[i + 1];
        else if (!strcmp(argv[i], "-n")) max_tokens = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-t")) temp       = (float)atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-k")) topp       = (float)atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-s")) seed       = (uint64_t)strtoull(argv[i + 1], NULL, 10);
        else if (!strcmp(argv[i], "-l")) max_seq    = atoi(argv[i + 1]);
    }

    /*
     * NVIDIA GPU + cuBLAS。重みは起動時に VRAM へアップロード。
     */
    gpu_print_device_info();

    char gpu_desc[256];
    gpu_get_device_desc(gpu_desc, sizeof gpu_desc);

    printf("Loading %s ...\n", model_path);

    Model model;
    memset(&model, 0, sizeof(model));

    model.fd = open(model_path, O_RDONLY);
    if (model.fd < 0) {
        fprintf(stderr, "Error: cannot open %s\n", model_path);
        return 1;
    }
    struct stat st;
    fstat(model.fd, &st);
    model.fsz = (size_t)st.st_size;
    model.fdata = (uint8_t *)mmap(NULL, model.fsz, PROT_READ, MAP_PRIVATE, model.fd, 0);
    if (model.fdata == MAP_FAILED) {
        fprintf(stderr, "Error: mmap failed\n");
        return 1;
    }

    char **merges = NULL;
    int n_merges = 0;
    parse_gguf(&model, &merges, &n_merges);

    model.cfg.max_seq = max_seq;

    Config *c = &model.cfg;
    printf("Model: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d\n",
           c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size);
    printf("       head_dim=%d n_rot=%d kv_dim=%d kv_mul=%d rope_theta=%g freq_scale=%g yarn_ext=%g attn_mscale=%g n_ctx_orig_yarn=%d max_seq=%d\n",
           c->head_dim, c->n_rot, c->kv_dim, c->kv_mul, (double)c->rope_theta,
           (double)c->rope_freq_scale, (double)c->yarn_ext_factor, (double)c->yarn_attn_factor,
           c->n_ctx_orig_yarn, c->max_seq);

    load_weights(&model);
    init_tokenizer(&model.tok, merges, n_merges);
    alloc_state(&model.s, c);

    GpuConfig gc = gpu_config_from(c);
    GpuWeightsHost gw = gpu_weights_from(&model.w);
    model.gpu = gpu_model_create(&gc, &gw);

    int n_prompt_tokens;
    int *prompt_tokens = chat_encode(&model.tok, prompt, &n_prompt_tokens);
    printf("Prompt: \"%s\" (%d tokens)\n\n", prompt, n_prompt_tokens);

    BenchLogInfo bench_meta;
    memset(&bench_meta, 0, sizeof bench_meta);
    bench_meta.model_path = model_path;
    bench_meta.prompt_text = prompt;
    bench_meta.gpu_desc = gpu_desc;
    bench_meta.max_new = max_tokens;
    bench_meta.temp = temp;
    bench_meta.topp = topp;
    bench_meta.seed = seed;
    bench_meta.max_seq = max_seq;
    generate(&model, prompt_tokens, n_prompt_tokens, max_tokens, temp, topp, seed,
             &bench_meta);

    free(prompt_tokens);
    gpu_model_destroy(model.gpu);
    free_state(&model.s);
    free_weight_ptrs(&model.w, c->n_layers);
    free(model.ti);
    free(model.tok.vocab);
    free(model.tok.vlen);
    free(model.tok.scores);
    free(model.tok.htab);
    munmap(model.fdata, model.fsz);
    close(model.fd);

    return 0;
}

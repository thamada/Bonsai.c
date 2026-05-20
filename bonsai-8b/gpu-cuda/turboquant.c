/*
 * TurboQuant CPU reference — PolarQuant (Lloyd-Max on Beta) + QJL residual.
 * Used to build tables at GPU init; device kernels mirror this logic.
 */

#include "turboquant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float tq_beta_pdf(float x, float alpha)
{
    float u = (x + 1.0f) * 0.5f;
    if (u < 1e-15f) u = 1e-15f;
    if (u > 1.0f - 1e-15f) u = 1.0f - 1e-15f;
    float t = 1.0f - u;
    float num = powf(u, alpha - 1.0f) * powf(t, alpha - 1.0f);
    float lg = lgammaf(alpha) + lgammaf(alpha) - lgammaf(2.0f * alpha);
    return num * expf(-lg) * 0.5f;
}

static void tq_lloyd_max(int dim, int bits, float *centroids, float *boundaries)
{
    const int n_levels = 1 << bits;
    const float alpha = (float)dim * 0.5f;

    for (int i = 0; i < n_levels; i++) {
        float p = ((float)i + 0.5f) / (float)n_levels;
        /* Approximate Beta(alpha,alpha) quantile via Wilson-Hilferty for speed */
        float z = sqrtf(2.0f) * ((i + 0.5f) / (float)n_levels - 0.5f) * 2.5f;
        float u = 0.5f * (1.0f + erff(z / sqrtf(2.0f)));
        if (u < 1e-6f) u = 1e-6f;
        if (u > 1.0f - 1e-6f) u = 1.0f - 1e-6f;
        centroids[i] = 2.0f * u - 1.0f;
        (void)p;
        (void)alpha;
    }

    for (int iter = 0; iter < 200; iter++) {
        boundaries[0] = -1.0f;
        boundaries[n_levels] = 1.0f;
        for (int i = 1; i < n_levels; i++)
            boundaries[i] = 0.5f * (centroids[i - 1] + centroids[i]);

        float max_shift = 0.0f;
        for (int i = 0; i < n_levels; i++) {
            float lo = boundaries[i], hi = boundaries[i + 1];
            float num = 0.0f, den = 0.0f;
            const int n_quad = 128;
            for (int q = 0; q < n_quad; q++) {
                float xs = lo + (hi - lo) * (float)q / (float)(n_quad - 1);
                float ps = tq_beta_pdf(xs, alpha);
                num += xs * ps;
                den += ps;
            }
            float nc = (den > 1e-15f) ? (num / den) : 0.5f * (lo + hi);
            float sh = fabsf(nc - centroids[i]);
            if (sh > max_shift) max_shift = sh;
            centroids[i] = nc;
        }
        if (max_shift < 1e-8f) break;
    }
    boundaries[0] = -1.0f;
    boundaries[n_levels] = 1.0f;
    for (int i = 1; i < n_levels; i++)
        boundaries[i] = 0.5f * (centroids[i - 1] + centroids[i]);
}

static uint32_t tq_xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float tq_randn(uint32_t *state)
{
    float u1 = (float)(tq_xorshift32(state) + 1) / 4294967296.0f;
    float u2 = (float)(tq_xorshift32(state) + 1) / 4294967296.0f;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void tq_gen_rotation(int dim, uint32_t seed, float *pi)
{
    float *a = (float *)malloc((size_t)dim * dim * sizeof(float));
    float *q = (float *)malloc((size_t)dim * dim * sizeof(float));
    float *r = (float *)malloc((size_t)dim * dim * sizeof(float));
    if (!a || !q || !r) {
        free(a); free(q); free(r);
        return;
    }
    uint32_t st = seed;
    for (int i = 0; i < dim * dim; i++)
        a[i] = tq_randn(&st);

    /* Modified Gram-Schmidt QR */
    for (int j = 0; j < dim; j++) {
        for (int i = 0; i < dim; i++)
            q[i * dim + j] = a[i * dim + j];
        for (int k = 0; k < j; k++) {
            float dot = 0.0f;
            for (int i = 0; i < dim; i++)
                dot += q[i * dim + j] * q[i * dim + k];
            for (int i = 0; i < dim; i++)
                q[i * dim + j] -= dot * q[i * dim + k];
        }
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += q[i * dim + j] * q[i * dim + j];
        norm = sqrtf(norm);
        if (norm < 1e-12f) norm = 1.0f;
        for (int i = 0; i < dim; i++)
            q[i * dim + j] /= norm;
    }
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            float s = 0.0f;
            for (int k = 0; k < dim; k++)
                s += q[k * dim + i] * a[k * dim + j];
            r[i * dim + j] = s;
        }
    }
    for (int j = 0; j < dim; j++) {
        float sign = (r[j * dim + j] >= 0.0f) ? 1.0f : -1.0f;
        for (int i = 0; i < dim; i++)
            pi[i * dim + j] = q[i * dim + j] * sign;
    }
    free(a);
    free(q);
    free(r);
}

static void tq_gen_qjl(int dim, uint32_t seed, float *s)
{
    uint32_t st = seed + 1;
    for (int i = 0; i < dim * dim; i++)
        s[i] = (tq_xorshift32(&st) & 1) ? 1.0f : -1.0f;
}

TurboQuantTables *turboquant_tables_create(int dim, int polar_bits, uint32_t seed)
{
    TurboQuantTables *tq = (TurboQuantTables *)calloc(1, sizeof(TurboQuantTables));
    if (!tq) return NULL;
    const int n_levels = 1 << polar_bits;
    tq->dim = dim;
    tq->polar_bits = polar_bits;
    tq->n_levels = n_levels;
    tq->centroids = (float *)malloc((size_t)n_levels * sizeof(float));
    tq->boundaries = (float *)malloc((size_t)(n_levels + 1) * sizeof(float));
    tq->pi = (float *)malloc((size_t)dim * dim * sizeof(float));
    tq->s = (float *)malloc((size_t)dim * dim * sizeof(float));
    if (!tq->centroids || !tq->boundaries || !tq->pi || !tq->s) {
        turboquant_tables_destroy(tq);
        return NULL;
    }
    tq_lloyd_max(dim, polar_bits, tq->centroids, tq->boundaries);
    tq_gen_rotation(dim, seed, tq->pi);
    tq_gen_qjl(dim, seed, tq->s);
    return tq;
}

void turboquant_tables_destroy(TurboQuantTables *tq)
{
    if (!tq) return;
    free(tq->centroids);
    free(tq->boundaries);
    free(tq->pi);
    free(tq->s);
    free(tq);
}

static int tq_quantize_coord(const TurboQuantTables *tq, float v)
{
    int idx = 0;
    while (idx < tq->n_levels - 1 && v >= tq->boundaries[idx + 1])
        idx++;
    return idx;
}

static void tq_pack_indices(const TurboQuantTables *tq, const int *idx, uint8_t *out)
{
    const int d = tq->dim;
    const int bits = tq->polar_bits;
    for (int i = 0; i < d; i++) {
        int shift = (i * bits) & 7;
        int byte_i = (i * bits) >> 3;
        out[byte_i] |= (uint8_t)((idx[i] & ((1 << bits) - 1)) << shift);
    }
}

static void tq_unpack_indices(const TurboQuantTables *tq, const uint8_t *packed, int *idx)
{
    const int d = tq->dim;
    const int bits = tq->polar_bits;
    const int mask = (1 << bits) - 1;
    for (int i = 0; i < d; i++) {
        int bitpos = i * bits;
        int byte_i = bitpos >> 3;
        int shift = bitpos & 7;
        idx[i] = (packed[byte_i] >> shift) & mask;
    }
}

static void tq_matvec(const float *m, const float *x, float *y, int dim)
{
    for (int i = 0; i < dim; i++) {
        float s = 0.0f;
        for (int j = 0; j < dim; j++)
            s += m[i * dim + j] * x[j];
        y[i] = s;
    }
}

void turboquant_compress(const TurboQuantTables *tq, const float *x,
                         TurboQuantPacked *out)
{
    const int d = tq->dim;
    float *y = (float *)malloc((size_t)d * sizeof(float));
    int *idx = (int *)malloc((size_t)d * sizeof(int));
    if (!y || !idx) {
        free(y);
        free(idx);
        return;
    }
    tq_matvec(tq->pi, x, y, d);
    for (int i = 0; i < d; i++)
        idx[i] = tq_quantize_coord(tq, y[i]);
    memset(out->indices, 0, (size_t)TQ_INDICES_BYTES(d, tq->polar_bits));
    memset(out->qjl, 0, (size_t)TQ_QJL_BYTES(d));
    tq_pack_indices(tq, idx, out->indices);

    float *res = (float *)malloc((size_t)d * sizeof(float));
    if (res) {
        for (int i = 0; i < d; i++)
            res[i] = y[i] - tq->centroids[idx[i]];
        float *proj = (float *)malloc((size_t)d * sizeof(float));
        if (proj) {
            tq_matvec(tq->s, res, proj, d);
            memset(out->qjl, 0, (size_t)TQ_QJL_BYTES(d));
            for (int i = 0; i < d; i++) {
                if (proj[i] >= 0.0f) {
                    int bi = i >> 3;
                    out->qjl[bi] |= (uint8_t)(1u << (i & 7));
                }
            }
            free(proj);
        }
        free(res);
    }
    free(y);
    free(idx);
}

float turboquant_inner_product(const TurboQuantTables *tq, const float *query,
                               const TurboQuantPacked *key)
{
    const int d = tq->dim;
    int *idx = (int *)malloc((size_t)d * sizeof(int));
    float *q_rot = (float *)malloc((size_t)d * sizeof(float));
    float *y_hat = (float *)malloc((size_t)d * sizeof(float));
    float *sq = (float *)malloc((size_t)d * sizeof(float));
    if (!idx || !q_rot || !y_hat || !sq) {
        free(idx); free(q_rot); free(y_hat); free(sq);
        return 0.0f;
    }

    tq_unpack_indices(tq, key->indices, idx);
    tq_matvec(tq->pi, query, q_rot, d);
    for (int i = 0; i < d; i++)
        y_hat[i] = tq->centroids[idx[i]];
    float base = 0.0f;
    for (int i = 0; i < d; i++)
        base += q_rot[i] * y_hat[i];

    tq_matvec(tq->s, q_rot, sq, d);
    float corr = 0.0f;
    const float scale = sqrtf((float)M_PI * 0.5f) / (float)d;
    for (int i = 0; i < d; i++) {
        int sign = (key->qjl[i >> 3] >> (i & 7)) & 1;
        float spm1 = sign ? 1.0f : -1.0f;
        corr += fabsf(sq[i]) * spm1;
    }
    free(idx);
    free(q_rot);
    free(y_hat);
    free(sq);
    return base + scale * corr;
}

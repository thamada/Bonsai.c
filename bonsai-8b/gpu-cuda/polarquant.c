/*
 * PolarQuant CPU reference — Lloyd-Max on Beta + random rotation Pi.
 * Device kernels in kernels.cu mirror this logic (no QJL / TurboQuant).
 */

#include "polarquant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float pq_beta_pdf(float x, float alpha)
{
    float u = (x + 1.0f) * 0.5f;
    if (u < 1e-15f) u = 1e-15f;
    if (u > 1.0f - 1e-15f) u = 1.0f - 1e-15f;
    float t = 1.0f - u;
    float num = powf(u, alpha - 1.0f) * powf(t, alpha - 1.0f);
    float lg = lgammaf(alpha) + lgammaf(alpha) - lgammaf(2.0f * alpha);
    return num * expf(-lg) * 0.5f;
}

static void pq_lloyd_max(int dim, int bits, float *centroids, float *boundaries)
{
    const int n_levels = 1 << bits;
    const float alpha = (float)dim * 0.5f;

    for (int i = 0; i < n_levels; i++) {
        float z = sqrtf(2.0f) * ((i + 0.5f) / (float)n_levels - 0.5f) * 2.5f;
        float u = 0.5f * (1.0f + erff(z / sqrtf(2.0f)));
        if (u < 1e-6f) u = 1e-6f;
        if (u > 1.0f - 1e-6f) u = 1.0f - 1e-6f;
        centroids[i] = 2.0f * u - 1.0f;
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
                float ps = pq_beta_pdf(xs, alpha);
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

static uint32_t pq_xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float pq_randn(uint32_t *state)
{
    float u1 = (float)(pq_xorshift32(state) + 1) / 4294967296.0f;
    float u2 = (float)(pq_xorshift32(state) + 1) / 4294967296.0f;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void pq_gen_rotation(int dim, uint32_t seed, float *pi)
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
        a[i] = pq_randn(&st);

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

PolarQuantTables *polarquant_tables_create(int dim, int polar_bits, uint32_t seed)
{
    PolarQuantTables *pq = (PolarQuantTables *)calloc(1, sizeof(PolarQuantTables));
    if (!pq) return NULL;
    const int n_levels = 1 << polar_bits;
    pq->dim = dim;
    pq->polar_bits = polar_bits;
    pq->n_levels = n_levels;
    pq->centroids = (float *)malloc((size_t)n_levels * sizeof(float));
    pq->boundaries = (float *)malloc((size_t)(n_levels + 1) * sizeof(float));
    pq->pi = (float *)malloc((size_t)dim * dim * sizeof(float));
    if (!pq->centroids || !pq->boundaries || !pq->pi) {
        polarquant_tables_destroy(pq);
        return NULL;
    }
    pq_lloyd_max(dim, polar_bits, pq->centroids, pq->boundaries);
    pq_gen_rotation(dim, seed, pq->pi);
    return pq;
}

void polarquant_tables_destroy(PolarQuantTables *pq)
{
    if (!pq) return;
    free(pq->centroids);
    free(pq->boundaries);
    free(pq->pi);
    free(pq);
}

static int pq_quantize_coord(const PolarQuantTables *pq, float v)
{
    int idx = 0;
    while (idx < pq->n_levels - 1 && v >= pq->boundaries[idx + 1])
        idx++;
    return idx;
}

static void pq_pack_indices(const PolarQuantTables *pq, const int *idx, uint8_t *out)
{
    const int d = pq->dim;
    const int bits = pq->polar_bits;
    for (int i = 0; i < d; i++) {
        int shift = (i * bits) & 7;
        int byte_i = (i * bits) >> 3;
        out[byte_i] |= (uint8_t)((idx[i] & ((1 << bits) - 1)) << shift);
    }
}

static void pq_unpack_indices(const PolarQuantTables *pq, const uint8_t *packed, int *idx)
{
    const int d = pq->dim;
    const int bits = pq->polar_bits;
    const int mask = (1 << bits) - 1;
    for (int i = 0; i < d; i++) {
        int bitpos = i * bits;
        int byte_i = bitpos >> 3;
        int shift = bitpos & 7;
        idx[i] = (packed[byte_i] >> shift) & mask;
    }
}

static void pq_matvec(const float *m, const float *x, float *y, int dim)
{
    for (int i = 0; i < dim; i++) {
        float s = 0.0f;
        for (int j = 0; j < dim; j++)
            s += m[i * dim + j] * x[j];
        y[i] = s;
    }
}

void polarquant_compress(const PolarQuantTables *pq, const float *x, uint8_t *out_indices)
{
    const int d = pq->dim;
    float *y = (float *)malloc((size_t)d * sizeof(float));
    int *idx = (int *)malloc((size_t)d * sizeof(int));
    if (!y || !idx) {
        free(y);
        free(idx);
        return;
    }
    pq_matvec(pq->pi, x, y, d);
    for (int i = 0; i < d; i++)
        idx[i] = pq_quantize_coord(pq, y[i]);
    memset(out_indices, 0, (size_t)PQ_INDICES_BYTES(d, pq->polar_bits));
    pq_pack_indices(pq, idx, out_indices);
    free(y);
    free(idx);
}

float polarquant_inner_product(const PolarQuantTables *pq, const float *query,
                               const uint8_t *key_indices)
{
    const int d = pq->dim;
    int *idx = (int *)malloc((size_t)d * sizeof(int));
    float *q_rot = (float *)malloc((size_t)d * sizeof(float));
    if (!idx || !q_rot) {
        free(idx);
        free(q_rot);
        return 0.0f;
    }
    pq_unpack_indices(pq, key_indices, idx);
    pq_matvec(pq->pi, query, q_rot, d);
    float s = 0.0f;
    for (int i = 0; i < d; i++)
        s += q_rot[i] * pq->centroids[idx[i]];
    free(idx);
    free(q_rot);
    return s;
}

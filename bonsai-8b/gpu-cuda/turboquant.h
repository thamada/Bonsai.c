#ifndef BONSAI_TURBOQUANT_H
#define BONSAI_TURBOQUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* TurboQuant (PolarQuant + QJL) for KV cache — arXiv:2504.19874, ICLR 2026.
 * Per head_dim vector: polar_bits per coordinate + 1-bit QJL residual. */

#define TQ_DIM_DEFAULT       128
#define TQ_POLAR_BITS_DEFAULT 2
#define TQ_N_LEVELS_DEFAULT  (1 << TQ_POLAR_BITS_DEFAULT)

/* Packed 2-bit indices: 4 coords / byte → dim/4 bytes */
#define TQ_INDICES_BYTES(d, bits)  (((d) * (bits) + 7) / 8)
/* QJL signs: 1 bit / dim */
#define TQ_QJL_BYTES(d)            (((d) + 7) / 8)

#define TQ_ENTRY_BYTES(d, bits) \
    (TQ_INDICES_BYTES(d, bits) + TQ_QJL_BYTES(d))

typedef struct {
    int dim;
    int polar_bits;
    int n_levels;
    float *centroids;   /* [n_levels] */
    float *boundaries;  /* [n_levels + 1] */
    float *pi;          /* [dim * dim] row-major */
    float *s;           /* [dim * dim] Rademacher QJL */
} TurboQuantTables;

typedef struct {
    uint8_t *indices;   /* packed polar indices, length TQ_INDICES_BYTES */
    uint8_t *qjl;       /* packed sign bits, length TQ_QJL_BYTES */
} TurboQuantPacked;

TurboQuantTables *turboquant_tables_create(int dim, int polar_bits, uint32_t seed);
void              turboquant_tables_destroy(TurboQuantTables *tq);

void turboquant_compress(const TurboQuantTables *tq, const float *x,
                         TurboQuantPacked *out);
/* V 用: Lloyd-Max のみ（QJL なし、indices のみ） */
void turboquant_compress_value(const TurboQuantTables *tq, const float *x,
                               uint8_t *out_indices);
float turboquant_inner_product(const TurboQuantTables *tq, const float *query,
                               const TurboQuantPacked *key);

#ifdef __cplusplus
}
#endif

#endif

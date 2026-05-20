#ifndef BONSAI_POLARQUANT_H
#define BONSAI_POLARQUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* PolarQuant KV cache — random rotation Pi + Lloyd-Max 2-bit (arXiv:2502.02617).
 * Per head_dim vector: polar_bits per coordinate in Pi-rotated space. */

#define PQ_DIM_DEFAULT       128
#define PQ_POLAR_BITS_DEFAULT 2
#define PQ_N_LEVELS_DEFAULT  (1 << PQ_POLAR_BITS_DEFAULT)

#define PQ_INDICES_BYTES(d, bits)  (((d) * (bits) + 7) / 8)
#define PQ_ENTRY_BYTES(d, bits)    PQ_INDICES_BYTES(d, bits)

typedef struct {
    int dim;
    int polar_bits;
    int n_levels;
    float *centroids;   /* [n_levels] */
    float *boundaries;  /* [n_levels + 1] */
    float *pi;          /* [dim * dim] row-major */
} PolarQuantTables;

PolarQuantTables *polarquant_tables_create(int dim, int polar_bits, uint32_t seed);
void              polarquant_tables_destroy(PolarQuantTables *pq);

void  polarquant_compress(const PolarQuantTables *pq, const float *x, uint8_t *out_indices);
float polarquant_inner_product(const PolarQuantTables *pq, const float *query,
                               const uint8_t *key_indices);

#ifdef __cplusplus
}
#endif

#endif

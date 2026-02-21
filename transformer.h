#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdint.h>

/* Cap sequence length to fit RunState in 520 KB SRAM */
#define MAX_SEQ_LEN 256

/* stories260K model dimensions — used for static buffer sizing */
#define MAX_DIM        64
#define MAX_HIDDEN_DIM 172
#define MAX_N_LAYERS   5
#define MAX_N_HEADS    8
#define MAX_N_KV_HEADS 4
#define MAX_VOCAB_SIZE 512
#define MAX_KV_DIM     ((MAX_DIM * MAX_N_KV_HEADS) / MAX_N_HEADS)  /* 32 */
#define MAX_HEAD_SIZE  (MAX_DIM / MAX_N_HEADS)                      /* 8 */

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    float *token_embedding_table;
    float *rms_att_weight;
    float *rms_ffn_weight;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *w1;
    float *w2;
    float *w3;
    float *rms_final_weight;
    float *wcls;
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;       /* points into key_cache */
    float *v;       /* points into value_cache */
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
} Transformer;

/**
 * Initialise the transformer: parse config from PSRAM, map weight pointers,
 * set up RunState to use static SRAM buffers. Returns 0 on success.
 */
int init_transformer(Transformer *t);

/**
 * Run one forward pass. Returns pointer to logits (vocab_size floats).
 */
float *forward(Transformer *t, int token, int pos);

#endif /* TRANSFORMER_H */

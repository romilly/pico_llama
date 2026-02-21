#include "transformer.h"
#include "psram.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/* ---- Static RunState buffers in SRAM (.bss) ---- */
static float rs_x[MAX_DIM];
static float rs_xb[MAX_DIM];
static float rs_xb2[MAX_DIM];
static float rs_hb[MAX_HIDDEN_DIM];
static float rs_hb2[MAX_HIDDEN_DIM];
static float rs_q[MAX_DIM];
static float rs_att[MAX_N_HEADS * MAX_SEQ_LEN];
static float rs_logits[MAX_VOCAB_SIZE];
static float rs_key_cache[MAX_N_LAYERS * MAX_SEQ_LEN * MAX_KV_DIM];
static float rs_value_cache[MAX_N_LAYERS * MAX_SEQ_LEN * MAX_KV_DIM];

/* ---- Weight pointer mapping ---- */

static void memory_map_weights(TransformerWeights *w, Config *p, float *ptr,
                               int shared_weights) {
    int head_size = p->dim / p->n_heads;
    int n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    /* skip freq_cis_real and freq_cis_imag */
    ptr += p->seq_len * head_size / 2;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

int init_transformer(Transformer *t) {
    /* Read config from the start of PSRAM (28-byte header) */
    Config *p = &t->config;
    memcpy(p, (void *)PSRAM_BASE, sizeof(Config));

    int shared_weights = p->vocab_size > 0 ? 1 : 0;
    p->vocab_size = p->vocab_size < 0 ? -p->vocab_size : p->vocab_size;

    printf("Transformer: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d "
           "vocab=%d seq_len=%d\n",
           p->dim, p->hidden_dim, p->n_layers, p->n_heads,
           p->n_kv_heads, p->vocab_size, p->seq_len);

    /* Validate against static buffer limits */
    if (p->dim > MAX_DIM || p->hidden_dim > MAX_HIDDEN_DIM ||
        p->n_layers > MAX_N_LAYERS || p->n_heads > MAX_N_HEADS ||
        p->n_kv_heads > MAX_N_KV_HEADS || p->vocab_size > MAX_VOCAB_SIZE) {
        printf("Transformer: ERROR — model exceeds static buffer sizes!\n");
        return -1;
    }

    /* Cap seq_len for KV cache sizing */
    if (p->seq_len > MAX_SEQ_LEN) {
        printf("Transformer: Capping seq_len from %d to %d\n",
               p->seq_len, MAX_SEQ_LEN);
        p->seq_len = MAX_SEQ_LEN;
    }

    /* Map weight pointers into PSRAM (after 28-byte / 7-int header) */
    float *weights_ptr = (float *)(PSRAM_BASE + sizeof(Config));
    memory_map_weights(&t->weights, p, weights_ptr, shared_weights);

    /* Wire RunState to static buffers */
    RunState *s = &t->state;
    s->x = rs_x;
    s->xb = rs_xb;
    s->xb2 = rs_xb2;
    s->hb = rs_hb;
    s->hb2 = rs_hb2;
    s->q = rs_q;
    s->att = rs_att;
    s->logits = rs_logits;
    s->key_cache = rs_key_cache;
    s->value_cache = rs_value_cache;
    s->k = NULL;
    s->v = NULL;

    printf("Transformer: Init OK (RunState in SRAM, weights in PSRAM)\n");
    return 0;
}

/* ---- Math helpers ---- */

static void rmsnorm(float *o, float *x, float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

static void matmul(float *xout, float *x, float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

/* ---- Forward pass ---- */

float *forward(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    /* Copy token embedding into x */
    float *content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));

    /* For each layer */
    for (int l = 0; l < p->n_layers; l++) {

        /* Attention rmsnorm */
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        /* KV cache pointers for this layer+position */
        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        /* QKV matmuls */
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        /* RoPE rotation */
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float *vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        /* Multi-head attention */
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_size;
            float *att = s->att + h * p->seq_len;

            for (int t = 0; t <= pos; t++) {
                float *k = s->key_cache + loff + t * kv_dim +
                           (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v = s->value_cache + loff + t * kv_dim +
                           (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        /* Output projection + residual */
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        /* FFN rmsnorm */
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        /* FFN: w1, w3, SiLU, w2 */
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        /* SiLU activation and element-wise multiply */
        for (int i = 0; i < hidden_dim; i++) {
            float v = s->hb[i];
            v *= (1.0f / (1.0f + expf(-v)));
            v *= s->hb2[i];
            s->hb[i] = v;
        }

        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        /* Residual */
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    /* Final rmsnorm */
    rmsnorm(x, x, w->rms_final_weight, dim);

    /* Classifier */
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

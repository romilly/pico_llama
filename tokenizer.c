#include "tokenizer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

/* Defined in model_data.h, included by main.c */
extern const unsigned char models_tok512_bin[];
extern const unsigned int models_tok512_bin_len;

/*
 * Static buffers — no malloc.
 * vocab_ptrs: array of char* pointing into the flash const array.
 * vocab_scores_buf: scores read from the binary.
 * sorted_vocab_buf: for BPE encode merge lookups.
 * str_buffer: scratch for encode().
 */
#define MAX_VOCAB 512

static char *vocab_ptrs[MAX_VOCAB];
static float vocab_scores_buf[MAX_VOCAB];
static TokenIndex sorted_vocab_buf[MAX_VOCAB];
static int sorted_vocab_ready = 0;
static char str_buffer[MAX_TOKEN_LENGTH * 2 + 3];

/*
 * We can't point directly into the packed binary for strings because they
 * aren't null-terminated there. So we copy each token string into a static
 * pool in SRAM.
 */
#define VOCAB_POOL_SIZE (MAX_VOCAB * (MAX_TOKEN_LENGTH + 1))
static char vocab_pool[VOCAB_POOL_SIZE];
static int vocab_pool_used = 0;

int init_tokenizer(Tokenizer *t, int vocab_size) {
    if (vocab_size > MAX_VOCAB) {
        printf("Tokenizer: vocab_size %d exceeds MAX_VOCAB %d\n",
               vocab_size, MAX_VOCAB);
        return -1;
    }

    t->vocab_size = vocab_size;
    t->vocab = vocab_ptrs;
    t->vocab_scores = vocab_scores_buf;
    t->sorted_vocab = NULL; /* built lazily on first encode() */

    /* Init byte_pieces for fallback single-byte decoding */
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    /* Parse the tok512.bin binary from flash */
    const unsigned char *ptr = models_tok512_bin;
    const unsigned char *end = models_tok512_bin + models_tok512_bin_len;

    /* First 4 bytes: max_token_length */
    if (ptr + 4 > end) return -2;
    memcpy(&t->max_token_length, ptr, sizeof(int));
    ptr += sizeof(int);

    printf("Tokenizer: max_token_length=%u, loading %d tokens...\n",
           t->max_token_length, vocab_size);

    vocab_pool_used = 0;

    for (int i = 0; i < vocab_size; i++) {
        /* score (float32) */
        if (ptr + sizeof(float) > end) return -3;
        memcpy(&vocab_scores_buf[i], ptr, sizeof(float));
        ptr += sizeof(float);

        /* len (int32) */
        int len;
        if (ptr + sizeof(int) > end) return -4;
        memcpy(&len, ptr, sizeof(int));
        ptr += sizeof(int);

        /* string (len bytes, not null-terminated in binary) */
        if (ptr + len > end) return -5;
        if (vocab_pool_used + len + 1 > VOCAB_POOL_SIZE) return -6;

        char *dest = vocab_pool + vocab_pool_used;
        memcpy(dest, ptr, len);
        dest[len] = '\0';
        vocab_ptrs[i] = dest;
        vocab_pool_used += len + 1;
        ptr += len;
    }

    printf("Tokenizer: Loaded %d tokens (%d bytes in pool)\n",
           vocab_size, vocab_pool_used);
    return 0;
}

char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    /* Strip leading space after BOS token */
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
    }
    printf("%s", piece);
}

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str, .id = 0 };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size,
                              sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos,
            int *tokens, int *n_tokens) {
    if (text == NULL) {
        printf("Tokenizer: cannot encode NULL text\n");
        *n_tokens = 0;
        return;
    }

    /* Build sorted vocab on first call */
    if (!sorted_vocab_ready) {
        for (int i = 0; i < t->vocab_size; i++) {
            sorted_vocab_buf[i].str = t->vocab[i];
            sorted_vocab_buf[i].id = i;
        }
        qsort(sorted_vocab_buf, t->vocab_size, sizeof(TokenIndex),
              compare_tokens);
        t->sorted_vocab = sorted_vocab_buf;
        sorted_vocab_ready = 1;
    }

    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    /* Add dummy prefix space */
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    /* Encode each character / UTF-8 codepoint */
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    /* BPE merge loop */
    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            snprintf(str_buffer, sizeof(str_buffer), "%s%s",
                     t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
}

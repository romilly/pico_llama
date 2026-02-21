#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>

#define MAX_TOKEN_LENGTH 128

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; /* 256 entries × 2 bytes */
} Tokenizer;

/**
 * Initialise tokenizer from the embedded tok512.bin const array in flash.
 * Returns 0 on success.
 */
int init_tokenizer(Tokenizer *t, int vocab_size);

/** Decode token id to string piece. */
char *decode(Tokenizer *t, int prev_token, int token);

/** Print a piece safely (skip non-printable single-byte tokens). */
void safe_printf(char *piece);

/**
 * Encode text into token ids.
 * tokens must have room for at least strlen(text)+3 entries.
 */
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos,
            int *tokens, int *n_tokens);

#endif /* TOKENIZER_H */

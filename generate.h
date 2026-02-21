#ifndef GENERATE_H
#define GENERATE_H

#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"

/**
 * Generate tokens from prompt. Streams output over USB serial and
 * reports tok/s at the end. steps=0 means use full seq_len.
 */
void generate(Transformer *transformer, Tokenizer *tokenizer,
              Sampler *sampler, char *prompt, int steps);

#endif /* GENERATE_H */

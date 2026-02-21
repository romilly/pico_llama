#include "generate.h"
#include <stdio.h>
#include <string.h>
#include "pico/time.h"

void generate(Transformer *transformer, Tokenizer *tokenizer,
              Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) prompt = empty_prompt;

    if (steps == 0 || steps > transformer->config.seq_len) {
        steps = transformer->config.seq_len;
    }

    /* Encode prompt — static buffer, max tokens = prompt length + 3 */
    int prompt_tokens[MAX_SEQ_LEN];
    int num_prompt_tokens = 0;
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        printf("Error: expected at least 1 prompt token\n");
        return;
    }

    printf("Prompt encoded to %d tokens\n", num_prompt_tokens);
    printf("Generating %d tokens...\n\n", steps);

    uint64_t start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < steps) {
        float *logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        /* BOS token = stop */
        if (next == 1) break;

        char *piece = decode(tokenizer, token, next);
        safe_printf(piece);
        /* Flush after each token for streaming effect */
        fflush(stdout);
        token = next;

        /* Start timing after first generated token */
        if (start == 0) start = time_us_64();
    }
    printf("\n");

    if (pos > 1) {
        uint64_t end = time_us_64();
        double elapsed_ms = (double)(end - start) / 1000.0;
        double toks = (pos - 1) / (elapsed_ms / 1000.0);
        printf("\n--- %d tokens in %.1f ms = %.1f tok/s ---\n",
               pos - 1, elapsed_ms, toks);
    }
}

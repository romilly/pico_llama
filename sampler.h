#ifndef SAMPLER_H
#define SAMPLER_H

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

/** Initialise sampler with static ProbIndex buffer. */
void init_sampler(Sampler *sampler, int vocab_size, float temperature,
                  float topp, unsigned long long rng_seed);

/** Sample next token from logits. */
int sample(Sampler *sampler, float *logits);

#endif /* SAMPLER_H */

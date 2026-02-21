#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#include "pico/time.h"
#include "psram.h"
#include "model_data.h"
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"
#include "generate.h"

static Transformer transformer;
static Tokenizer tokenizer;
static Sampler sampler;

static int load_model_to_psram(void) {
    size_t psram_avail = psram_size();
    if (psram_avail == 0) {
        printf("Model: No PSRAM, cannot load.\n");
        return -1;
    }

    printf("Model: Copying %u bytes from flash to PSRAM...\n",
           (unsigned)models_stories260K_bin_len);

    if (models_stories260K_bin_len > psram_avail) {
        printf("Model: Too large for PSRAM! (%u > %u)\n",
               (unsigned)models_stories260K_bin_len, (unsigned)psram_avail);
        return -2;
    }

    absolute_time_t t0 = get_absolute_time();
    memcpy((void *)PSRAM_BASE, models_stories260K_bin, models_stories260K_bin_len);
    absolute_time_t t1 = get_absolute_time();
    int64_t copy_us = absolute_time_diff_us(t0, t1);
    printf("Model: Copy done in %lld ms (%.1f MB/s)\n",
           copy_us / 1000, (double)models_stories260K_bin_len / copy_us);

    return 0;
}

int main(void) {
    stdio_init_all();
    sleep_ms(5000);

    if (cyw43_arch_init()) {
        printf("CYW43 init failed!\n");
        return 1;
    }

    printf("\n=== Pico LLaMA ===\n\n");

    /* PSRAM init */
    printf("Initialising PSRAM...\n");
    int psram_rc = psram_setup();
    if (psram_rc == 0) {
        printf("PSRAM: Init OK — %u MB\n", (unsigned)(psram_size() >> 20));
    } else {
        printf("PSRAM: Init failed (rc=%d)\n", psram_rc);
        return 1;
    }

    /* Copy model weights to PSRAM */
    if (load_model_to_psram() != 0) {
        printf("Failed to load model to PSRAM\n");
        return 1;
    }

    /* Init transformer (maps weights from PSRAM, sets up RunState in SRAM) */
    if (init_transformer(&transformer) != 0) {
        printf("Failed to init transformer\n");
        return 1;
    }

    /* Init tokenizer from embedded flash data */
    if (init_tokenizer(&tokenizer, transformer.config.vocab_size) != 0) {
        printf("Failed to init tokenizer\n");
        return 1;
    }

    /* Init sampler: temperature=1.0, topp=0.9, seed from timer */
    unsigned long long rng_seed = (unsigned long long)time_us_64();
    init_sampler(&sampler, transformer.config.vocab_size, 1.0f, 0.9f, rng_seed);

    printf("\n=== Generating ===\n\n");

    /* Generate a story */
    generate(&transformer, &tokenizer, &sampler, "Once upon a time", 256);

    /* Blink LED to show we're alive */
    printf("\n=== Done — blinking LED ===\n");
    while (1) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        sleep_ms(500);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
        sleep_ms(500);
    }

    return 0;
}

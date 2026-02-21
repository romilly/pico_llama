#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#include "pico/time.h"
#include "psram.h"
#include "model_data.h"

/* LLaMA-2 model config header: 7 x int32 */
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

static int model_rc = -1;
static Config model_config = { 0 };

static void load_model(void) {
    size_t psram_avail = psram_size();
    if (psram_avail == 0) {
        printf("Model: No PSRAM, cannot load.\n");
        model_rc = -1;
        return;
    }

    printf("Model: Copying %u bytes from flash to PSRAM...\n",
           (unsigned)models_stories260K_bin_len);

    if (models_stories260K_bin_len > psram_avail) {
        printf("Model: Too large for PSRAM! (%u > %u)\n",
               (unsigned)models_stories260K_bin_len, (unsigned)psram_avail);
        model_rc = -2;
        return;
    }

    absolute_time_t t0 = get_absolute_time();
    memcpy((void *)PSRAM_BASE, models_stories260K_bin, models_stories260K_bin_len);
    absolute_time_t t1 = get_absolute_time();
    int64_t copy_us = absolute_time_diff_us(t0, t1);
    printf("Model: Copy done in %lld ms (%.1f MB/s)\n",
           copy_us / 1000, (double)models_stories260K_bin_len / copy_us);

    /* Parse config header from PSRAM */
    memcpy(&model_config, (void *)PSRAM_BASE, sizeof(Config));

    printf("Model: dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, "
           "n_kv_heads=%d, vocab_size=%d, seq_len=%d\n",
           model_config.dim, model_config.hidden_dim,
           model_config.n_layers, model_config.n_heads,
           model_config.n_kv_heads, model_config.vocab_size,
           model_config.seq_len);

    /* Sanity check */
    if (model_config.dim == 64 && model_config.n_layers == 5 &&
        model_config.vocab_size == 512) {
        printf("Model: Header looks correct!\n");
        model_rc = 0;
    } else {
        printf("Model: WARNING — header values unexpected\n");
        model_rc = -3;
    }
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
    }

    /* Load model weights into PSRAM */
    load_model();

    printf("\n=== Running ===\n");
    int count = 0;
    while (1) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        printf("tick %d | PSRAM: %u MB rc=%d | Model: rc=%d dim=%d layers=%d vocab=%d\n",
               count++,
               (unsigned)(psram_size() >> 20), psram_rc,
               model_rc, model_config.dim, model_config.n_layers,
               model_config.vocab_size);
        sleep_ms(500);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
        sleep_ms(500);
    }

    return 0;
}

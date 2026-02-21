/* psram.h - PSRAM init for Pico Plus 2W
   Based on fanpico psram.c by Timo Kokkonen (GPL-3.0-or-later) */

#ifndef PSRAM_H
#define PSRAM_H 1

#include <stdint.h>
#include <stddef.h>
#include "hardware/platform_defs.h"
#include "hardware/regs/addressmap.h"

#define PSRAM_BASE         _u(0x11000000)
#define PSRAM_NOCACHE_BASE _u(0x15000000)
#define PSRAM_WINDOW_SIZE  (16 << 20)

#define PSRAM_CS_PIN 47

typedef struct psram_id_t {
    uint8_t mfid;
    uint8_t kgd;
    uint8_t eid[6];
} psram_id_t;

/* Init PSRAM in QSPI mode. Returns 0 on success, negative on error. */
int psram_setup(void);

/* Return detected PSRAM size in bytes (0 if not initialised). */
size_t psram_size(void);

/* Return pointer to PSRAM chip ID (NULL if not initialised). */
const psram_id_t* psram_get_id(void);

#endif

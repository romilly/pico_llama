/* psram.c - PSRAM init for Pico Plus 2W (APS6404L, QSPI via QMI)
   Based on fanpico psram.c by Timo Kokkonen (GPL-3.0-or-later) */

#include <stdio.h>
#include <string.h>
#include "hardware/structs/qmi.h"
#include "hardware/structs/xip_ctrl.h"
#include "hardware/gpio.h"
#include "hardware/clocks.h"
#include "hardware/sync.h"

#include "psram.h"

/* PSRAM commands */
#define CMD_READ_ID           0x9f
#define CMD_QUAD_READ         0xeb
#define CMD_QUAD_WRITE        0x38
#define CMD_ENTER_QPI_MODE    0x35
#define CMD_EXIT_QPI_MODE     0xf5
#define CMD_RESET_ENABLE      0x66
#define CMD_RESET             0x99

/* KGD (Known Good Die) */
#define KGD_PASS              0x5d

/* AP Memory APS6404L timings */
#define APMEMORY_MAX_CLK      109000000
#define APMEMORY_MAX_SELECT_FS 8000000000ULL
#define APMEMORY_MIN_DESELECT_FS 18000000ULL

#define PSRAM_MAX_CSR_CLK     5000000

static size_t psram_sz = 0;
static psram_id_t psram_id_data = { 0 };


static inline void csr_busy_wait(void)
{
    while (qmi_hw->direct_csr & QMI_DIRECT_CSR_BUSY_BITS)
        tight_loop_contents();
}

static inline void csr_txempty_wait(void)
{
    while ((qmi_hw->direct_csr & QMI_DIRECT_CSR_TXEMPTY_BITS) == 0)
        tight_loop_contents();
}

static inline void csr_enable_direct_mode(uint8_t csr_clkdiv)
{
    qmi_hw->direct_csr = csr_clkdiv << QMI_DIRECT_CSR_CLKDIV_LSB | QMI_DIRECT_CSR_EN_BITS;
    csr_busy_wait();
}

static inline void csr_disable_direct_mode(void)
{
    hw_clear_bits(&qmi_hw->direct_csr, QMI_DIRECT_CSR_EN_BITS | QMI_DIRECT_CSR_ASSERT_CS1N_BITS);
}

static void __no_inline_not_in_flash_func(csr_send_command)(uint32_t cmd)
{
    hw_set_bits(&qmi_hw->direct_csr, QMI_DIRECT_CSR_ASSERT_CS1N_BITS);
    qmi_hw->direct_tx = cmd;
    csr_txempty_wait();
    csr_busy_wait();
    (void)qmi_hw->direct_rx;
    hw_clear_bits(&qmi_hw->direct_csr, QMI_DIRECT_CSR_ASSERT_CS1N_BITS);
}

static void __no_inline_not_in_flash_func(psram_read_id)(uint8_t csr_clkdiv, uint8_t *buffer)
{
    uint32_t saved_ints = save_and_disable_interrupts();

    csr_enable_direct_mode(csr_clkdiv);

    /* Make sure chip is in SPI mode */
    csr_send_command(
        QMI_DIRECT_TX_OE_BITS |
        QMI_DIRECT_TX_IWIDTH_VALUE_Q << QMI_DIRECT_TX_IWIDTH_LSB |
        CMD_EXIT_QPI_MODE);

    /* Send Read ID command */
    hw_set_bits(&qmi_hw->direct_csr, QMI_DIRECT_CSR_ASSERT_CS1N_BITS);
    for (int i = 0; i < (4 + 8); i++) {
        qmi_hw->direct_tx = (i == 0 ? CMD_READ_ID : 0x00);
        csr_txempty_wait();
        csr_busy_wait();
        uint8_t rx = qmi_hw->direct_rx;
        if (i >= 4)
            buffer[i - 4] = rx;
    }

    csr_disable_direct_mode();
    restore_interrupts(saved_ints);
}

static void __no_inline_not_in_flash_func(psram_qmi_setup)(uint8_t clkdiv, uint8_t csr_clkdiv,
                                                             uint8_t max_select, uint8_t min_deselect,
                                                             uint8_t rxdelay)
{
    uint32_t saved_ints = save_and_disable_interrupts();

    /* Reset and enter QPI mode */
    csr_enable_direct_mode(csr_clkdiv);
    csr_send_command(CMD_RESET_ENABLE);
    csr_send_command(CMD_RESET);
    csr_send_command(CMD_ENTER_QPI_MODE);
    csr_disable_direct_mode();

    /* Configure QMI M1 timing */
    qmi_hw->m[1].timing = (
        1                                  << QMI_M1_TIMING_COOLDOWN_LSB |
        QMI_M1_TIMING_PAGEBREAK_VALUE_1024 << QMI_M1_TIMING_PAGEBREAK_LSB |
        0                                  << QMI_M1_TIMING_SELECT_SETUP_LSB |
        3                                  << QMI_M1_TIMING_SELECT_HOLD_LSB |
        max_select                         << QMI_M1_TIMING_MAX_SELECT_LSB |
        min_deselect                       << QMI_M1_TIMING_MIN_DESELECT_LSB |
        rxdelay                            << QMI_M1_TIMING_RXDELAY_LSB |
        clkdiv                             << QMI_M1_TIMING_CLKDIV_LSB
    );

    /* Configure read format: quad-width everything */
    qmi_hw->m[1].rfmt = (
        QMI_M1_RFMT_DUMMY_LEN_VALUE_24    << QMI_M1_RFMT_DUMMY_LEN_LSB |
        QMI_M1_RFMT_SUFFIX_LEN_VALUE_NONE << QMI_M1_RFMT_SUFFIX_LEN_LSB |
        QMI_M1_RFMT_PREFIX_LEN_VALUE_8    << QMI_M1_RFMT_PREFIX_LEN_LSB |
        QMI_M1_RFMT_DATA_WIDTH_VALUE_Q    << QMI_M1_RFMT_DATA_WIDTH_LSB |
        QMI_M1_RFMT_DUMMY_WIDTH_VALUE_Q   << QMI_M1_RFMT_DUMMY_WIDTH_LSB |
        QMI_M1_RFMT_SUFFIX_WIDTH_VALUE_Q  << QMI_M1_RFMT_SUFFIX_WIDTH_LSB |
        QMI_M1_RFMT_ADDR_WIDTH_VALUE_Q    << QMI_M1_RFMT_ADDR_WIDTH_LSB |
        QMI_M1_RFMT_PREFIX_WIDTH_VALUE_Q  << QMI_M1_RFMT_PREFIX_WIDTH_LSB
    );
    qmi_hw->m[1].rcmd = CMD_QUAD_READ << QMI_M1_RCMD_PREFIX_LSB;

    /* Configure write format: quad-width, no dummy cycles */
    qmi_hw->m[1].wfmt = (
        QMI_M1_WFMT_DUMMY_LEN_VALUE_NONE  << QMI_M1_WFMT_DUMMY_LEN_LSB |
        QMI_M1_WFMT_SUFFIX_LEN_VALUE_NONE << QMI_M1_WFMT_SUFFIX_LEN_LSB |
        QMI_M1_WFMT_PREFIX_LEN_VALUE_8    << QMI_M1_WFMT_PREFIX_LEN_LSB |
        QMI_M1_WFMT_DATA_WIDTH_VALUE_Q    << QMI_M1_WFMT_DATA_WIDTH_LSB |
        QMI_M1_WFMT_DUMMY_WIDTH_VALUE_Q   << QMI_M1_WFMT_DUMMY_WIDTH_LSB |
        QMI_M1_WFMT_SUFFIX_WIDTH_VALUE_Q  << QMI_M1_WFMT_SUFFIX_WIDTH_LSB |
        QMI_M1_WFMT_ADDR_WIDTH_VALUE_Q    << QMI_M1_WFMT_ADDR_WIDTH_LSB |
        QMI_M1_WFMT_PREFIX_WIDTH_VALUE_Q  << QMI_M1_WFMT_PREFIX_WIDTH_LSB
    );
    qmi_hw->m[1].wcmd = CMD_QUAD_WRITE << QMI_M1_WCMD_PREFIX_LSB;

    restore_interrupts(saved_ints);

    /* Enable writes to PSRAM memory window */
    hw_set_bits(&xip_ctrl_hw->ctrl, XIP_CTRL_WRITABLE_M1_BITS);
}

int psram_setup(void)
{
    uint32_t sys_clk = clock_get_hz(clk_sys);
    uint64_t clock_period_fs = 1000000000000000ULL / sys_clk;
    uint8_t csr_clkdiv = (sys_clk + PSRAM_MAX_CSR_CLK - 1) / PSRAM_MAX_CSR_CLK;

    psram_sz = 0;

    /* Configure CS1 GPIO */
    gpio_set_function(PSRAM_CS_PIN, GPIO_FUNC_XIP_CS1);

    /* Read chip ID */
    psram_read_id(csr_clkdiv, (uint8_t *)&psram_id_data);
    if (psram_id_data.kgd != KGD_PASS) {
        printf("PSRAM: No chip detected (KGD=0x%02x)\n", psram_id_data.kgd);
        return -1;
    }

    /* Determine size from density field (AP Memory encoding) */
    uint8_t density = (psram_id_data.eid[0] >> 5);
    psram_sz = 2;
    if (density == 1) psram_sz = 4;
    else if (density == 2) psram_sz = 8;
    psram_sz <<= 20;  /* convert MB to bytes */

    /* Calculate clock divider */
    uint8_t clkdiv = (sys_clk + APMEMORY_MAX_CLK - 1) / APMEMORY_MAX_CLK;

    /* Calculate timings */
    uint8_t max_select = (APMEMORY_MAX_SELECT_FS >> 6) / clock_period_fs;
    uint8_t min_deselect = (APMEMORY_MIN_DESELECT_FS + clock_period_fs - 1) / clock_period_fs;
    uint8_t rxdelay = 1 + (sys_clk > 150000000 ? clkdiv : 1);

    /* Enable QSPI mode */
    psram_qmi_setup(clkdiv, csr_clkdiv, max_select, min_deselect, rxdelay);

    /* Quick write test */
    volatile uint32_t *psram = (volatile uint32_t *)PSRAM_NOCACHE_BASE;
    psram[0] = 0xdeadc0de;
    if (psram[0] != 0xdeadc0de) {
        printf("PSRAM: Write test failed!\n");
        psram_sz = 0;
        return -2;
    }
    psram[0] = 0;

    return 0;
}

size_t psram_size(void)
{
    return psram_sz;
}

const psram_id_t *psram_get_id(void)
{
    return &psram_id_data;
}

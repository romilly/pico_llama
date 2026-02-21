# Project Brief: LLM Inference on Pimoroni Pico Plus 2W

## Goal

Port Karpathy's **llama2.c** to run on a **Pimoroni Pico Plus 2W** microcontroller board, generating text from a tiny LLaMA-2 model using the board's PSRAM to store weights. This is an experimental/educational project — the aim is to get tokens streaming over USB serial, not to achieve production performance.

## Target Hardware

**Board:** Pimoroni Pico Plus 2W (PIM726)

| Resource | Spec |
|---|---|
| SoC | RP2350B (80-pin variant) |
| CPU | Dual Arm Cortex-M33 @ 150 MHz |
| On-chip SRAM | 520 KB |
| PSRAM | 8 MB (APS6404L, QSPI via QMI) |
| Flash | 16 MB QSPI |
| Wireless | RM2 module (2.4 GHz WiFi + BT 5.2) |
| USB | USB-C (power + programming) |
| User LED | GP25 |
| BOOT button | GP45 (active low, usable as user button) |
| Debug | 3-pin JST-SH SWD connector |

**Pico SDK board identifier:** `pimoroni_pico_plus2_w_rp2350`

## Source Code

**Repository:** https://github.com/karpathy/llama2.c

The key file is `run.c` — a single-file, zero-dependency, pure C implementation of LLaMA-2 inference. It uses only `<stdio.h>`, `<stdlib.h>`, `<math.h>`, `<string.h>`, `<time.h>`. No ML frameworks, no BLAS, no threads required.

## Model Weights

From Karpathy's `tinyllamas` collection on Hugging Face (`huggingface.co/karpathy/tinyllamas`):

| Model | dim | layers | heads | kv_heads | context | params | .bin size (fp32) |
|---|---|---|---|---|---|---|---|
| **stories260K** | 64 | 5 | 8 | 4 | 512 | 260K | ~1 MB |
| stories15M | 288 | 6 | 6 | 6 | 256 | 15M | ~58 MB |
| stories42M | 512 | 8 | 8 | 8 | 1024 | 42M | ~168 MB |

**Primary target: `stories260K`** — fits comfortably in PSRAM (~1 MB weights). This model uses a custom vocab of 512 tokens and a matching tokenizer (`tok512.bin`).

**Stretch goal: `stories15M`** — at ~58 MB in fp32, this exceeds PSRAM. Would require int8 quantization (llama2.c has a `runq.c` quantised variant) to bring it to ~15 MB, which fits in PSRAM. Use `stories15M_q80.bin` if available, or quantise via the repo's export tooling.

Download URLs:
- `https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin`
- `https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin`
- `https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin`

The `.bin` format is simple: a 28-byte header (7 × int32: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len), followed by fp32 weights in a fixed order. The tokenizer `.bin` is similarly straightforward (see `run.c` for the reader).

## Architecture Overview

### Memory Layout

```
0x10000000  Flash (16 MB) — code + embedded model weights
0x11000000  PSRAM (8 MB, cached XIP) — model weights at runtime
0x15000000  PSRAM (8 MB, uncached) — same physical PSRAM, different window
0x20000000  SRAM (520 KB) — stack, run state, KV cache, hot buffers
```

### Strategy

1. **Weights in PSRAM** — Load model weights (from flash or embedded binary blob) into PSRAM at startup. The RP2350's XIP cache means sequential reads through weights during matmul will benefit from burst transfers.

2. **Run state in SRAM** — The `RunState` struct (activations, KV cache, scratch buffers) should live in fast SRAM. For `stories260K`, the run state is small enough (~tens of KB). For larger models, the KV cache may need to spill to PSRAM.

3. **Dual-core** — Use both M33 cores for matrix multiplication. The matmul in `run.c` is a simple loop that's trivially parallelisable by splitting rows across cores. Use the Pico SDK's `multicore_launch_core1()` and a shared flag/spinlock.

### Key Adaptations Needed from run.c

- **Remove `mmap`/file I/O** — `run.c` uses `fopen`/`fread` or `mmap` to load weights. On Pico there's no filesystem. Embed the `.bin` as a const array in flash (via `xxd -i` or `objcopy`), or store it raw in flash and read from a known address.

- **Replace `malloc` with static/placed allocations** — Use `__attribute__((section(".psram")))` for weight buffers, or manually assign pointers to PSRAM addresses. Standard `malloc` on Pico SDK only allocates from the 520 KB SRAM heap.

- **Replace `printf` with USB serial** — Use `stdio_init_all()` and `printf` will go to USB CDC automatically with Pico SDK.

- **Replace `time()` with Pico timer** — Use `time_us_64()` for tok/s measurement.

- **Remove `system()` and other POSIX calls** — These don't exist on bare-metal.

## PSRAM Initialisation

The PSRAM chip on the Pico Plus 2W is connected to GPIO 47 (CS1). A minimal init (SPI mode, slow) is:

```c
#include "hardware/gpio.h"
#include "hardware/structs/xip_ctrl.h"

void init_psram_basic() {
    gpio_set_function(47, GPIO_FUNC_XIP_CS1);
    xip_ctrl_hw->ctrl |= XIP_CTRL_WRITABLE_M1_BITS;
}
// PSRAM is now accessible at 0x11000000 (cached) or 0x15000000 (uncached)
```

**However**, this leaves PSRAM in slow SPI mode. For decent bandwidth, you need to configure QSPI (quad mode) via the QMI registers. This involves:

1. Enter direct CSR mode and issue the PSRAM's QPI-enable command (0x35)
2. Configure M1 timing registers (`qmi_hw->m[1].timing`) for the clock divider and page boundary
3. Configure M1 read/write formats and commands for quad-width transfers (`qmi_hw->m[1].rfmt`, `qmi_hw->m[1].rcmd`, `qmi_hw->m[1].wfmt`, `qmi_hw->m[1].wcmd`)

A well-tested reference implementation lives in the `fanpico` project: `https://github.com/tjko/fanpico/blob/main/src/psram.c` — this handles detection, QPI enable, and timing configuration for the APS6404L chip used on the Pico Plus 2W. The SparkFun pico library (`https://github.com/sparkfun/sparkfun-pico`) also has PSRAM detection and init code with a malloc-like allocator.

Running the RP2350 at 200 MHz with a QMI clock divider of 2 gives PSRAM at 100 MHz, which is within spec for the APS6404L at 3.3V. Measured sequential bandwidth at this config is roughly 10+ MB/s.

### Custom Linker Script

To place data in PSRAM, add a section to a custom linker script:

```ld
MEMORY {
    FLASH(rx)     : ORIGIN = 0x10000000, LENGTH = 16M
    RAM(rwx)      : ORIGIN = 0x20000000, LENGTH = 512k
    SCRATCH_X(rwx): ORIGIN = 0x20080000, LENGTH = 4k
    SCRATCH_Y(rwx): ORIGIN = 0x20081000, LENGTH = 4k
    PSRAM(rwx)    : ORIGIN = 0x11000000, LENGTH = 8M
}

SECTIONS {
    .psram (NOLOAD) : {
        . = ALIGN(4);
        *(.psram*)
    } > PSRAM
}
```

Then in C:
```c
__attribute__((section(".psram"))) uint8_t model_weights[MODEL_SIZE];
```

Or simply cast the PSRAM base address:
```c
#define PSRAM_BASE ((uint8_t*)0x11000000)
// memcpy weights from flash to PSRAM_BASE at startup
```

## Build Setup

### Prerequisites

- Pico SDK 2.x (https://github.com/raspberrypi/pico-sdk)
- `arm-none-eabi-gcc` toolchain
- CMake 3.13+

### CMakeLists.txt Skeleton

```cmake
cmake_minimum_required(VERSION 3.13)

set(PICO_BOARD pimoroni_pico_plus2_w_rp2350)

include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(pico_llama C CXX ASM)

pico_sdk_init()

add_executable(pico_llama
    main.c
)

target_link_libraries(pico_llama
    pico_stdlib
    pico_multicore
    hardware_gpio
    hardware_clocks
)

# USB serial output
pico_enable_stdio_usb(pico_llama 1)
pico_enable_stdio_uart(pico_llama 0)

# Generate UF2 for drag-and-drop flashing
pico_add_extra_outputs(pico_llama)

# Optional: custom linker script for PSRAM section
# pico_set_linker_script(pico_llama ${CMAKE_SOURCE_DIR}/memmap_psram.ld)
```

### Flashing

Hold BOOT button while plugging in USB-C. The board mounts as a USB mass storage device. Copy the `.uf2` file onto it. It auto-reboots and runs.

## Expected Performance

For the 260K model on a single core at 150 MHz:

- The main bottleneck is matrix multiplication: `dim × dim` and `dim × hidden_dim` operations with dim=64
- With dim=64, these are tiny matmuls — should complete very quickly
- **Estimated: 50–200+ tokens/sec** (the model is extremely small)
- The 15M model (if quantised to int8) would be more memory-bandwidth-bound: **estimated 2–10 tok/s** depending on PSRAM bandwidth utilisation

For reference, the ESP32-S3 (240 MHz, 2 MB PSRAM) achieves 19 tok/s on the 260K model using dual cores and SIMD.

## Implementation Steps

1. **Get a basic Pico SDK project building and flashing** — blink LED, printf over USB serial. Confirm toolchain works.

2. **Initialise and test PSRAM** — Write a pattern, read it back, report over USB serial. Ideally configure QSPI mode for full bandwidth.

3. **Embed the stories260K model** — Either use `xxd -i stories260K.bin > model_data.h` to create a C array in flash, or use CMake's `target_link_options` to place the raw binary at a known flash offset.

4. **Port run.c** — Strip out file I/O, mmap, and POSIX dependencies. Replace memory allocation with PSRAM pointers. Keep the core transformer forward pass untouched.

5. **Get single-core inference working** — Stream tokens over USB serial.

6. **Add dual-core matmul** — Split the `matmul` function across both cores for a potential ~1.5–2× speedup.

7. **Benchmark and optimise** — Measure tok/s, identify bottlenecks (likely PSRAM bandwidth for larger models), try overclocking to 200 MHz.

## Potential Pitfalls

- **PSRAM is much slower than SRAM for random access** — sequential burst reads are fast, but scattered access patterns pay a heavy penalty (16+ QSPI clock cycles overhead per access). The XIP cache helps, but cache thrashing on weight matrices larger than the cache will hurt.

- **No hardware float unit for single-precision on M33** — actually, the Cortex-M33 in RP2350 *does* have an FPU (single-precision), so fp32 matmul is fine. The RP2350 also has some DSP/SIMD instructions that could help with fixed-point.

- **Flash size for larger models** — the 15M model at 58 MB fp32 won't fit in 16 MB flash. You'd need the int8 quantised version (~15 MB) or stream from SD card.

- **USB serial buffer** — `printf` over USB CDC can block if the host isn't reading. Consider a ring buffer or checking tud_cdc_connected().

## Key References

- llama2.c source: https://github.com/karpathy/llama2.c
- Pico SDK docs: https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html
- RP2350 datasheet (QMI/PSRAM chapter 12.14): https://datasheets.raspberrypi.com/rp2350/rp2350-datasheet.pdf
- PSRAM init reference (fanpico): https://github.com/tjko/fanpico/blob/main/src/psram.c
- PSRAM init reference (sparkfun-pico): https://github.com/sparkfun/sparkfun-pico
- Pimoroni board header for SDK: `pimoroni_pico_plus2_w_rp2350` (included in Pico SDK 2.x)
- PSRAM forum thread: https://forums.raspberrypi.com/viewtopic.php?t=375109
- PSRAM overclocking gist: https://gist.github.com/eightycc/b61813c05899281ce7d2a2f86490be3b

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Port of Karpathy's **llama2.c** to run LLaMA-2 inference on a **Pimoroni Pico Plus 2W** (RP2350B, dual Cortex-M33 @ 150 MHz, 8 MB PSRAM, 16 MB flash). The primary target model is `stories260K` (~1 MB weights, fits in PSRAM).

The project briefing is in `pico-llm-briefing.md` — read it for hardware details, memory layout, PSRAM init, and implementation strategy.

## Build System

This is a C project using the **Pico SDK 2.x** and CMake. The board identifier is `pimoroni_pico_plus2_w_rp2350`.

### Prerequisites

- Pico SDK 2.x (`$PICO_SDK_PATH` must be set)
- `arm-none-eabi-gcc` toolchain
- CMake 3.13+

### Build Commands

```bash
mkdir -p build && cd build
cmake -DPICO_BOARD=pimoroni_pico_plus2_w_rp2350 ..
make -j$(nproc)
```

### Flashing

Hold BOOT button while plugging in USB-C. Copy the `.uf2` file from `build/` to the mounted USB mass storage device.

## Architecture

- **Weights** live in PSRAM (0x11000000, cached XIP window)
- **RunState** (activations, KV cache, scratch buffers) lives in SRAM (0x20000000, 520 KB)
- **Model binary** embedded in flash, copied to PSRAM at startup
- **Output** streamed as tokens over USB CDC serial

### Key Adaptations from llama2.c

- No filesystem — model weights embedded as const array or placed at known flash offset
- No `mmap`/`malloc` for weights — use PSRAM section attributes or direct address casting
- USB serial via `stdio_init_all()` / `printf`
- Timing via `time_us_64()` instead of `time()`
- Dual-core matmul via `multicore_launch_core1()` for parallelism

## Key References

- Source being ported: https://github.com/karpathy/llama2.c (`run.c`)
- PSRAM init reference: https://github.com/tjko/fanpico/blob/main/src/psram.c
- RP2350 datasheet (QMI/PSRAM in chapter 12.14): https://datasheets.raspberrypi.com/rp2350/rp2350-datasheet.pdf

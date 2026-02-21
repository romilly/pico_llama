# Pico LLaMA

LLaMA-2 inference running on a **Pimoroni Pico Plus 2W** microcontroller. A port of Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) to bare-metal RP2350.

## What It Does

Generates text from a tiny LLaMA-2 model entirely on-device, streaming tokens over USB serial. No host computer needed after flashing -- just plug in and watch it write stories.

Currently running the `stories260K` model (~1 MB weights) at **19 tokens/second**.

### Example Output

```
Once upon a time, there was a little boy named Timmy. Timmy loved to sleep
at the park with his friends. One day, Timmy's friend Billy came over to
play...
```

## Hardware

**Board:** [Pimoroni Pico Plus 2W](https://shop.pimoroni.com/products/pico-plus-2-w) (PIM726)

| Resource       | Spec                                  |
|----------------|---------------------------------------|
| SoC            | RP2350B (dual Cortex-M33 @ 150 MHz)  |
| SRAM           | 520 KB                                |
| PSRAM          | 8 MB (APS6404L, QSPI)                |
| Flash          | 16 MB QSPI                           |
| Connectivity   | WiFi + BT 5.2 (unused for inference) |
| Interface      | USB-C (power + serial output)         |

## Memory Layout

- **Flash (16 MB):** Firmware + model binary embedded as const array
- **PSRAM (8 MB):** Model weights copied here at startup (cached XIP window at `0x11000000`)
- **SRAM (520 KB):** RunState buffers -- activations, KV cache, scratch space

## Prerequisites

- [Pico SDK 2.x](https://github.com/raspberrypi/pico-sdk)
- `arm-none-eabi-gcc` toolchain
- CMake 3.13+

## Building

```bash
export PICO_SDK_PATH=/path/to/pico-sdk
mkdir -p build && cd build
cmake -DPICO_BOARD=pimoroni_pico_plus2_w_rp2350 ..
make -j$(nproc)
```

This produces `build/pico_llama.uf2`.

## Flashing

1. Hold the **BOOT** button while plugging in USB-C
2. Copy the UF2 to the mounted drive:

```bash
cp build/pico_llama.uf2 /media/$USER/RP2350/
```

The board reboots automatically and begins generating.

## Monitoring Output

The Pico streams tokens over USB CDC serial. To watch (reconnects automatically on reboot):

```bash
while true; do cat /dev/ttyACM0 2>/dev/null; sleep 0.1; done
```

You'll see the startup banner, model config, then generated text followed by a tok/s measurement. The onboard LED blinks when generation is complete.

## Project Structure

```
main.c            -- Entry point: init hardware, load model, generate
transformer.c/h   -- Forward pass: matmul, attention, FFN, RoPE
tokenizer.c/h     -- BPE tokenizer (vocabulary embedded in flash)
sampler.c/h       -- Temperature scaling, top-p sampling
generate.c/h      -- Token generation loop with timing
psram.c/h         -- PSRAM init via QMI (RP2350-specific)
model_data.h      -- Declares embedded model binary (in models/)
CMakeLists.txt    -- Build config targeting Pico SDK 2.x
```

## Key Adaptations from llama2.c

The original `run.c` assumes a desktop environment with filesystem, `mmap`, `malloc`, and `time.h`. This port replaces all of that:

- **No filesystem** -- model weights embedded as a const array in flash, copied to PSRAM at boot
- **No dynamic allocation** -- all buffers are statically sized in `.bss` (SRAM)
- **USB serial** -- `stdio_init_all()` / `printf` over CDC
- **Timing** -- `time_us_64()` instead of `time()`
- **PSRAM** -- custom QMI init for the APS6404L chip on the Pico Plus 2W

## Performance

| Model        | Tokens/sec | Notes                  |
|--------------|------------|------------------------|
| stories260K  | 19         | Single core, 150 MHz   |

## License

GPL-3.0 -- see [LICENSE](LICENSE).

This project is GPL-3.0 because `psram.c`/`psram.h` are derived from [fanpico](https://github.com/tjko/fanpico) by Timo Kokkonen, which is GPL-3.0-or-later. The goal is to eventually replace this with an independent PSRAM implementation, allowing re-licensing under MIT.

## Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for llama2.c and the tinyllamas models
- [Pimoroni](https://pimoroni.com/) for the Pico Plus 2W board
- [Raspberry Pi](https://www.raspberrypi.com/) for the RP2350 and Pico SDK

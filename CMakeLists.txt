cmake_minimum_required(VERSION 3.13)

set(PICO_BOARD pimoroni_pico_plus2_w_rp2350)

include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(pico_llama C CXX ASM)

pico_sdk_init()

add_executable(pico_llama
    main.c
    psram.c
    transformer.c
    tokenizer.c
    sampler.c
    generate.c
)

target_link_libraries(pico_llama
    pico_stdlib
    pico_cyw43_arch_none
    pico_time
    hardware_gpio
    hardware_clocks
)

# USB serial output
pico_enable_stdio_usb(pico_llama 1)
pico_enable_stdio_uart(pico_llama 0)

# Generate UF2 for drag-and-drop flashing
pico_add_extra_outputs(pico_llama)

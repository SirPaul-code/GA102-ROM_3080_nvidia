cmake_minimum_required(VERSION 3.24)
project(ga102_rom LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# This project is intentionally compiled for Ampere GA10x only.
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)

add_executable(ga102-rom src/main.cu)

target_compile_options(ga102-rom PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

if(MSVC)
    target_compile_options(ga102-rom PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
else()
    target_compile_options(ga102-rom PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

set_target_properties(ga102-rom PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

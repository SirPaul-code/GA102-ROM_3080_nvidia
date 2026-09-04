cmake_minimum_required(VERSION 3.24)
project(ga102_rom LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# This project is intentionally compiled for Ampere GA10x only.
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)

add_executable(ga102-rom src/main.cu)
add_executable(ga102-rom-v2 src/bench_v2.cu)
add_executable(ga102-rom-v3 src/bench_v3.cu)
add_executable(ga102-rom-v4 src/bench_v4.cu)
add_executable(ga102-rom-v5 src/bench_v5.cu)
add_executable(ga102-rom-v6 src/bench_v6.cu)
add_executable(ga102-rom-v7 src/bench_v7.cu)
add_executable(ga102-rom-v8 src/bench_v8.cu)
add_executable(ga102-rom-v9 src/bench_v9.cu)
add_executable(ga102-rom-v10 src/bench_v10.cu)
add_executable(ga102-rom-v11 src/bench_v11.cu)
add_executable(ga102-rom-v12 src/bench_v12.cu)
add_executable(ga102-rom-v13 src/bench_v13.cu)

set(GA102_PREINCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/src/preinclude.cuh")

target_compile_options(ga102-rom PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo;--pre-include=${GA102_PREINCLUDE}>
)

target_compile_options(ga102-rom-v2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

target_compile_options(ga102-rom-v3 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

target_compile_options(ga102-rom-v4 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

target_compile_options(ga102-rom-v5 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

target_compile_options(ga102-rom-v6 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

target_compile_options(ga102-rom-v7 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

target_compile_options(ga102-rom-v8 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

target_compile_options(ga102-rom-v9 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

target_compile_options(ga102-rom-v10 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

target_compile_options(ga102-rom-v11 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

target_compile_options(ga102-rom-v12 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

target_compile_options(ga102-rom-v13 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

# CUDA 13.x no longer exposes CUDART_INF_F through the headers used by V8.
# The attention reduction only needs a finite sentinel below every real score.
target_compile_definitions(ga102-rom-v8 PRIVATE CUDART_INF_F=3.402823466e+38F)

if(MSVC)
    target_compile_options(ga102-rom PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ga102-rom-v2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ga102-rom-v3 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ga102-rom-v4 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ga102-rom-v5 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ga102-rom-v6 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ga102-rom-v7 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ga102-rom-v8 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ga102-rom-v9 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ga102-rom-v10 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ga102-rom-v11 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ga102-rom-v12 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ga102-rom-v13 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
else()
    target_compile_options(ga102-rom PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(ga102-rom-v2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(ga102-rom-v3 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(ga102-rom-v4 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(ga102-rom-v5 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(ga102-rom-v6 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(ga102-rom-v7 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(ga102-rom-v8 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(ga102-rom-v9 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(ga102-rom-v10 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(ga102-rom-v11 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(ga102-rom-v12 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(ga102-rom-v13 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

set_target_properties(ga102-rom PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

set_target_properties(ga102-rom-v2 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

set_target_properties(ga102-rom-v3 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

set_target_properties(ga102-rom-v4 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

set_target_properties(ga102-rom-v5 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

set_target_properties(ga102-rom-v6 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

set_target_properties(ga102-rom-v7 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

set_target_properties(ga102-rom-v8 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

set_target_properties(ga102-rom-v9 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

set_target_properties(ga102-rom-v10 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

set_target_properties(ga102-rom-v11 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

set_target_properties(ga102-rom-v12 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

set_target_properties(ga102-rom-v13 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 86
)

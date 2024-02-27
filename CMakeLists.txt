cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
### set(CMAKE_CUDA_ARCHITECTURES "52;75")
set (CMAKE_CC_COMPILER /usr/bin/clang)
set (CMAKE_CXX_COMPILER /usr/bin/clang++)
set (CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc )
project(cmake_and_cuda LANGUAGES CXX CUDA)
#
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
#
set (CMAKE_CUDA_FLAGS "-std=c++17 -Xptxas -v -D_FORCE_INLINES -DLOG_LEVEL=1 -DGPU_PROFILE") # "-gencode arch=compute_53,code=sm_53"],

# add cuda toolkit files
include_directories(/usr/local/cuda/include)

# RUN:
# export CC=/usr/bin/clang
# export CXX=/usr/bin/clang++
# CUDACXX=/usr/local/cuda/bin/nvcc cmake -GNinja ..
# OR
# cmake -GNinja ..

include(CTest)

add_library(wamca_cc STATIC
    source/log.cpp
    source/except.cpp
    source/mtrand.cpp
    source/tsplib.cpp
    source/utils.cpp
    source/log.h
    source/except.h
    source/mtrand.h
    source/tsplib.h
    source/utils.h
    source/types.h
    source/graph.hpp
    source/thirdparty/UnpackTuple.hpp
    source/consts.h
  )

add_library(wamca_cuda STATIC
    source/mlk2opt.cu
    source/mlkswap.cu
    source/mlkoropt.cu
    source/mlkernel.cu
    source/mlsolution.cpp # SHOULD BE .cu
    source/mlads.cpp
    source/mlproblem.cpp
    source/mlk2opt.h
    source/gpu.h # SHOULD BE .cuh
    source/cuda/gpu_helper.h
    source/cuda/gpu_string.h
    source/mlkernel.h
    source/WamcaExperiment.hpp
    source/mlads.h
    source/mlsolution.h
    source/mlproblem.h
    source/mlkswap.h
    source/mlkoropt.h
)

set_target_properties( wamca_cuda
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(main_wamca source/main.cpp)

# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set_target_properties(wamca_cuda PROPERTIES CUDA_ARCHITECTURES "52;75")
set_target_properties(main_wamca PROPERTIES CUDA_ARCHITECTURES "52;75")

set_property(TARGET main_wamca 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main_wamca PRIVATE wamca_cuda wamca_cc)

if(APPLE) # why this fix? TODO: test!
  set_property(TARGET main_wamca 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
cmake_minimum_required(VERSION 3.20)
project(galaxy C CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(galaxy galaxy.cpp ParticleRenderer.cpp galaxy_kernel.cu)

set(CMAKE_CUDA_SAMPLES_INCLUDE_DIRECTORIES "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1/common/inc")

set(CMAKE_CUDA_SAMPLES_INCLUDE_DIRECTORIES_2 "C:/Users/juani/Desktop/ComputerGraphics/GPU Computing/CUDA/cuda-samples/Common")


message("CUDA TOOLKIT INCLUDE: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message("CUDA SAMPLES INCLUDE: " ${CMAKE_CUDA_SAMPLES_INCLUDE_DIRECTORIES})

target_include_directories(galaxy PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        "${CMAKE_CUDA_SAMPLES_INCLUDE_DIRECTORIES}"
        "${CMAKE_CUDA_SAMPLES_INCLUDE_DIRECTORIES_2}"
        )

if((MSCV) OR (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    target_link_directories(
            galaxy PUBLIC
            lib/freeglut-3.0.0MSVC/x64
            lib/glew
    )
endif()

target_link_libraries(galaxy
        "freeglut.lib"
        "glew32.lib")

#set_target_properties(default PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)

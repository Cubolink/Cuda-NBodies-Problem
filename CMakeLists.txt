cmake_minimum_required(VERSION 3.20)
project(nbodies-problem C CXX)

# Check for CUDA availability
find_package(CUDA)

if(CUDA_FOUND)
  enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 14)
  set(USE_CUDA ON)
else()
  set(USE_CUDA OFF)
endif()

set(PARTICLE_RENDERER_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/source/particle-renderer")
set(PARTICLE_RENDERER_LIBRARY particle-renderer)

add_subdirectory(source)
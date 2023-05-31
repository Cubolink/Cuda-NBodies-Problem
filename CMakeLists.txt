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

set(NBODIES_UTIL_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/source/nbodies-util")
set(NBODIES_UTIL_LIBRARY nbodies-util)

add_subdirectory(source)
cmake_minimum_required(VERSION 3.20)
project(nbodies-problem C CXX)

# Check for CUDA availability
find_package(CUDA)

if(CUDA_FOUND)
  message(">>> CUDA found, building project with it")
  enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 14)
  set(USE_CUDA ON)
else()
  message(">>> CUDA was not found")
  set(USE_CUDA OFF)
endif()

add_subdirectory(source)
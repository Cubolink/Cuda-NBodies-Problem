# Check if CUDA is available
find_package(CUDA QUIET)

# Configure CUDA CPP project if CUDA is found
if(CUDA_FOUND)
  add_executable(nbodies-cuda nbodies-cuda.cu)

  target_include_directories(nbodies-cuda PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    ${NBODIES_UTIL_INCLUDE_DIRECTORIES}
  )

  target_link_libraries(nbodies-cuda ${NBODIES_UTIL_LIBRARY})
else()
  # CUDA not found, handle the absence
  message(WARNING ">> CUDA not found. Skipping CUDA NBody implementation")
endif()
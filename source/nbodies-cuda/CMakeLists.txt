# Check if CUDA is available
find_package(CUDA QUIET)

# Configure CUDA CPP project if CUDA is found
if(CUDA_FOUND)
  add_executable(nbodies-cuda nbodies-cuda.cpp nbodies-kernel.cu)

  target_include_directories(nbodies-cuda PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    ${NBODIES_UTIL_INCLUDE_DIRECTORIES}
  )

  if((MSCV) OR (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    target_link_directories(nbodies-cuda PUBLIC
      "${PROJECT_SOURCE_DIR}/lib/freeglut-3.0.0MSVC/x64"
      "${PROJECT_SOURCE_DIR}/lib/glew"
    )
  endif()

  target_link_libraries(nbodies-cuda "freeglut.lib" "glew32.lib" ${NBODIES_UTIL_LIBRARY})
else()
  # CUDA not found, handle the absence
  message(WARNING ">> CUDA not found. Skipping CUDA NBody implementation")
endif()
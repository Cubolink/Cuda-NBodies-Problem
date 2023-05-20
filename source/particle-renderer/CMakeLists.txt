add_library(particle-renderer STATIC ParticleRenderer.cpp ParticleRenderer.h)

set_property(TARGET particle-renderer PROPERTY CXX_STANDARD 20)
target_include_directories(particle-renderer PRIVATE "${PROJECT_SOURCE_DIR}/include")

if((MSCV) OR (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
  target_link_directories(particle-renderer PUBLIC
    "${PROJECT_SOURCE_DIR}/lib/freeglut-3.0.0MSVC/x64"
    "${PROJECT_SOURCE_DIR}/lib/glew"
  )
endif()

target_link_libraries(particle-renderer "freeglut.lib" "glew32.lib")
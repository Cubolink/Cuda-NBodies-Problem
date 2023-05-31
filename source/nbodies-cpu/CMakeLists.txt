add_executable(nbodies-cpu nbodies-cpu.cpp)

set_property(TARGET nbodies-cpu PROPERTY CXX_STANDARD 20)
target_include_directories(nbodies-cpu PRIVATE "${PROJECT_SOURCE_DIR}/include" ${NBODIES_UTIL_INCLUDE_DIRECTORIES})

if((MSCV) OR (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
  target_link_directories(nbodies-cpu PUBLIC
    "${PROJECT_SOURCE_DIR}/lib/freeglut-3.0.0MSVC/x64"
    "${PROJECT_SOURCE_DIR}/lib/glew"
  )
endif()

target_link_libraries(nbodies-cpu "freeglut.lib" "glew32.lib" ${NBODIES_UTIL_LIBRARY})
add_library(nbodies-util STATIC
            camera.cpp
            camera.h
            controller.cpp
            controller.h
            data-loader.cpp
            data-loader.h
            particle-renderer.cpp 
            particle-renderer.h
            particle-timer.cpp
            particle-timer.h)

set_property(TARGET nbodies-util PROPERTY CXX_STANDARD 20)
target_include_directories(nbodies-util PRIVATE "${PROJECT_SOURCE_DIR}/include")

if((MSCV) OR (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
  target_link_directories(nbodies-util PUBLIC
    "${PROJECT_SOURCE_DIR}/lib/freeglut-3.0.0MSVC/x64"
    "${PROJECT_SOURCE_DIR}/lib/glew"
  )
endif()

target_link_libraries(nbodies-util "freeglut.lib" "glew32.lib")
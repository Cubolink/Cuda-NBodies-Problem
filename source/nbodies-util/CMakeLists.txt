add_library(nbodies-util STATIC
            data-loader.cpp
            data-loader.h
            particle-timer.cpp
            particle-timer.h)

set_property(TARGET nbodies-util PROPERTY CXX_STANDARD 20)
target_include_directories(nbodies-util PRIVATE "${PROJECT_SOURCE_DIR}/include")

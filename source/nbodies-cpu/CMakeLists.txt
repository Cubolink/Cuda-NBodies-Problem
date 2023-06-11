add_executable(nbodies-cpu nbodies-cpu.cpp)

set_property(TARGET nbodies-cpu PROPERTY CXX_STANDARD 20)
target_include_directories(nbodies-cpu PRIVATE "${PROJECT_SOURCE_DIR}/include" ${NBODIES_UTIL_INCLUDE_DIRECTORIES})


target_link_libraries(nbodies-cpu ${NBODIES_UTIL_LIBRARY})
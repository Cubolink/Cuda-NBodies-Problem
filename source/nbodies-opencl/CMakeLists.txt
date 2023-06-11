# Check if CUDA is available
find_package(OpenCL REQUIRED)

if(OpenCL_FOUND)
    # Configure OpenCL CPP project if OpenCL is found
    add_executable(nbodies-opencl nbodies-opencl.cpp)

    target_include_directories(nbodies-opencl PUBLIC
            ${PROJECT_SOURCE_DIR}/include
            ${NBODIES_UTIL_INCLUDE_DIRECTORIES}
    )

    target_link_libraries(nbodies-opencl ${NBODIES_UTIL_LIBRARY} OpenCL::OpenCL)
else()
    # OpenCL not found
    message(WARNING  ">> OpenCL not found. Skipping OpenCL NBody implementation")
endif()
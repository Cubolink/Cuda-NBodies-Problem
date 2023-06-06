# Check if CUDA is available
find_package(OpenCL REQUIRED)

if(OpenCL_FOUND)
    # Configure OpenCL CPP project if OpenCL is found
    add_executable(nbodies-opencl nbodies-opencl.cpp)

    target_include_directories(nbodies-opencl PUBLIC
            ${PROJECT_SOURCE_DIR}/include
            #${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            ${NBODIES_UTIL_INCLUDE_DIRECTORIES}
    )

    if((MSCV) OR (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
        target_link_directories(nbodies-opencl PUBLIC
                ${PROJECT_SOURCE_DIR}/lib/freeglut-3.0.0MSVC/x64
                ${PROJECT_SOURCE_DIR}/lib/glew
        )
    endif()

    target_link_libraries(nbodies-opencl "freeglut.lib" "glew32.lib" ${NBODIES_UTIL_LIBRARY} OpenCL::OpenCL)
else()
    # OpenCL not found
    message(WARNING  ">> OpenCL not found. Skipping OpenCL NBody implementation")
endif()
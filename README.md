# Cuda-NBodies-Problem

**This project** contains implementations to solve the N-Body Problem using three different computing architectures:
* CPU
* CUDA
* OpenCL

**This problem** is a physics problems, where we want to model the motions of each particle in a group of n-particles which
interact gravitationally with each other. This means we have to compute the force between each pair of particles in order
to get the total acceleration and movements of each particle.

**The Goal** is to compare the performance of these different implementations, using different GPU block sizes, memory,
problem's grid size, measuring the speed-ups.

## Build & Compile Instructions

### Building with CMake

You can use CMake to build this project. To do so, make sure you create a `build` folder. You can follow these steps:

```console
mkdir build
cd build
cmake ..
```

You may need a specific CUDA-compatible compiler, so you may need to select a generator when building with cmake.
For example,
```console
cd build
cmake .. -G "Visual Studio 15 2017 Win64" 
```
Please, refer to `cmake --help` to see your available generators, make sure you use the same as your compiler.

### Compiling the project

Once you built the CMake project, you can go to your build folder and compile it. Some generators create a makefile,
so you can run
```console
make
```

If you are using a Visual Studio compiler, you can open the solution file with your Visual Studio IDE and compile it from there.

If you want to use the MSBuild compiler from terminal, you can run
```console
msbuild nbodies-problem.sln
```

### Running the Project

In your build folder, you will have the CPU, CUDA and OpenCL implementations inside a `your_build_folder/source` folder.
If you are using the Visual Studio IDE, you can open the solution (that you opened to compile) and run the projects from there.
If you used another method, or used msbuild, you will have to copy some folders first:
* Copy the `data` folder (which contains shaders and galaxy data) into `your_build` folder.
* Copy the `source/nbodies-opencl/kernel.cl` file of the project, into `your_build/source/nbodies-opencl`.

#### Changing the experiment variables

We have provided some macros you can comment/uncomment in order to change the parameters.

The CPU, CUDA and OpenCL implementations provide the following macro you can change.
```
// Number of particles to be loaded from file
#define NUM_BODIES 16384
```

The following macros are only available in CUDA and OpenCL.
```
// Block size
#define BLOCK_SIZE 256  // GROUP_SIZE in OpenCL
```

Make sure you only define one of the following macros.
```
// Memory configuration
#define GLOBAL_MEMORY
// #define LOCAL_MEMORY
```

Same here.

```
// Block configuration
#define ONE_DIM_BLOCK
// #define TWO_DIM_BLOCK
```

### Running without OpenGL

You may want to run the experiments without displaying the galaxies with OpenGL.
For this, you can check out the `main-closed-gl` branch where we have provided the same implementations, but removing the
OpenGL dependency.
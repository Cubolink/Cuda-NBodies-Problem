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

If you are using a Visual Studio compiler, you can run
```console
msbuild galaxy.sln
```
Or open the solution file with your Visual Studio IDE and compile it from there.
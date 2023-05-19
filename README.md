# Cuda-NBodies-Problem

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
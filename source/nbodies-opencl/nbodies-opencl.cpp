//
// Created by Cubolink on 31-05-2023.
//

#include <iostream>
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

int main()
{
    std::cout << "Hello OpenCL project!" << std::endl;
}
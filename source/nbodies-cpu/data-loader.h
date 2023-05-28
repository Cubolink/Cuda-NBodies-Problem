//
// Created by major on 20-05-2023.
//

#ifndef NBODIES_PROBLEM_DATA_LOADER_H
#define NBODIES_PROBLEM_DATA_LOADER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "data-structs.h"

void loadData(const char* filename, int bodies, float3 *positions, float3 *velocities, float *masses, float scaleFactor)
{
    int skip = 49152 / bodies;

    std::ifstream fin(filename);
    if (!fin) {
        std::cout << "Error reading the file " << filename;
        exit(0);
    }

    std::string line;
    int i = 0;
    bool end_file = false;
    while(i < bodies)
    {
        for (int j = 0; j < skip; j++)
        {
            if(!std::getline(fin, line, '\n'))
                end_file = true;

            if (end_file)
                break;
        }
        if (end_file)
            break;

        std::stringstream ss(line);
        std::string item;


        // Position
        std::getline(ss, item, ' ');
        positions[i].x = std::stof(item) * scaleFactor;
        std::getline(ss, item, ' ');
        positions[i].y = std::stof(item) * scaleFactor;
        std::getline(ss, item, ' ');
        positions[i].z = std::stof(item) * scaleFactor;

        // Velocity
        std::getline(ss, item, ' ');
        velocities[i].x = std::stof(item);
        std::getline(ss, item, ' ');
        velocities[i].y = std::stof(item);
        std::getline(ss, item, ' ');
        velocities[i].z = std::stof(item);

        i++;
    }
}

#endif //NBODIES_PROBLEM_DATA_LOADER_H

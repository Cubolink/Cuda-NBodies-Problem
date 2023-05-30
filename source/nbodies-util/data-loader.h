// =================================
// Created by Cubolink on 20-05-2023
// =================================

#ifndef NBODIES_PROBLEM_DATA_LOADER_H
#define NBODIES_PROBLEM_DATA_LOADER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

void loadData(const char* filename, int bodies, float* positions, float* velocities, float* masses, float positionFactor)
{
    // Attributes factors
    const float velocityFactor = 8.0f;
    const float massFactor = 120000.0f;

    // Particle skip
    int skip = 49152 / bodies;

    std::ifstream fin(filename);
    if (!fin) {
        std::cout << "Error reading the file " << filename;
        exit(0);
    }

    std::string line;
    int i = 0;
    while(i < bodies)
    {
        for (int j = 0; j < skip; j++)
            if (!std::getline(fin, line, '\n')) return;

        std::stringstream ss(line);
        std::string item;

        // Mass
        std::getline(ss, item, ' ');
        masses[i] = std::stof(item) * massFactor;

        // Position
        std::getline(ss, item, ' ');
        positions[3 * i] = std::stof(item) * positionFactor;
        std::getline(ss, item, ' ');
        positions[3 * i + 1] = std::stof(item) * positionFactor;
        std::getline(ss, item, ' ');
        positions[3 * i + 2] = std::stof(item) * positionFactor;

        // Velocity
        std::getline(ss, item, ' ');
        velocities[3 * i] = std::stof(item) * velocityFactor;
        std::getline(ss, item, ' ');
        velocities[3 * i + 1] = std::stof(item) * velocityFactor;
        std::getline(ss, item, ' ');
        velocities[3 * i + 2] = std::stof(item) * velocityFactor;

        i++;
    }
}

#endif //NBODIES_PROBLEM_DATA_LOADER_H

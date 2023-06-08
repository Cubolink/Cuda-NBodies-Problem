// ==================================
// Created by Juanxpeke on 01-06-2023
// ==================================

#include "particle-timer.h"
#include <string>
#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>

ParticleTimer::ParticleTimer(int nParticles)
: m_numParticles(nParticles),
  m_totalElapsedTime(0.0),
  m_iterationCount(0)
{
}

void ParticleTimer::startIteration()
{
    m_startTime = std::chrono::high_resolution_clock::now();
}

void ParticleTimer::endIteration()
{
    m_endTime = std::chrono::high_resolution_clock::now();
    double elapsedTime = std::chrono::duration<double>(m_endTime - m_startTime).count();
    m_totalElapsedTime += elapsedTime;
    m_iterationCount++;

    double averageElapsedTime = m_totalElapsedTime / m_iterationCount;
    store["iterations"].push_back(std::make_pair(m_totalElapsedTime, m_iterationCount));
    store["particles_per_second"].push_back(std::make_pair(m_totalElapsedTime, m_numParticles / averageElapsedTime));
}

void ParticleTimer::printParticleEvaluatedPerSecond()
{
    double averageElapsedTime = m_totalElapsedTime / m_iterationCount;
    double particlesPerSecond = m_numParticles / averageElapsedTime;
    std::cout << "Particles evaluated per second: " << particlesPerSecond << std::endl;
}

void ParticleTimer::exportData(const std::string &folderPath) {
    std::cout << "Exporting "<< std::endl;

    std::filesystem::create_directories(folderPath);

    // For each vector of the store
    for (auto it = store.begin(); it != store.end(); ++it)
    {
        std::string key = it->first;
        std::vector<std::pair<double, double>> values = it->second;

        // Write into a representative csv file all the pairs from the vector
        std::string filePath = folderPath + key + ".csv";
        std::ofstream file(filePath);
        for (auto vIt = values.begin(); vIt != values.end(); ++vIt)
        {
            double p1 = vIt->first;
            double p2 = vIt->second;
            if (file.is_open()) {
                file << p1 << "," << p2 << std::endl;
            } else
                std::cout << "Failed to open the file: " << filePath << std::endl;
        }
        file.close();
    }
}

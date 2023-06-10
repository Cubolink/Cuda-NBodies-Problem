// ==================================
// Created by Juanxpeke on 01-06-2023
// ==================================

#include "particle-timer.h"
#include <string>
#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>

ParticleTimer::ParticleTimer(int nParticles, const std::string &dataFolderName)
: m_numParticles(nParticles),
  m_totalElapsedTime(0.0),
  m_iterationCount(0),
  m_dataExported(false),
  m_dataFolderName(dataFolderName)
{
}

void ParticleTimer::startIteration()
{
    m_startTime = std::chrono::high_resolution_clock::now();
}

void ParticleTimer::endIteration()
{
    if (m_iterationCount >= 100 && !m_dataExported) {
        exportData("data-" + m_dataFolderName + "/");
        m_dataExported = true;
        return;
    } else if (m_dataExported) {
        return;
    }
        
    m_endTime = std::chrono::high_resolution_clock::now();
    double elapsedTime = std::chrono::duration<double>(m_endTime - m_startTime).count();
    m_totalElapsedTime += elapsedTime;
    m_iterationCount++;

    double particlesPerSecond = m_numParticles / elapsedTime;

    store["iterations"].push_back(std::make_pair(m_totalElapsedTime, m_iterationCount));
    store["particles-per-second"].push_back(std::make_pair(m_totalElapsedTime, particlesPerSecond));

    std::cout << "Speed " << m_iterationCount - 1 << " (Particles per second): " << particlesPerSecond << std::endl;
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

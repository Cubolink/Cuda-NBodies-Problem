// ==================================
// Created by Juanxpeke on 01-06-2023
// ==================================

#include "particle-timer.h"
#include <iostream>
#include <chrono>

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
}

void ParticleTimer::printParticleEvaluatedPerSecond()
{
    double averageElapsedTime = m_totalElapsedTime / m_iterationCount;
    double particlesPerSecond = m_numParticles / averageElapsedTime;
    std::cout << "Particles evaluated per second: " << particlesPerSecond << std::endl;
}
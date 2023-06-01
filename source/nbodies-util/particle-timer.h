// ==================================
// Created by Juanxpeke on 01-06-2023
// ==================================

#ifndef __PARTICLETIMER_H__
#define __PARTICLETIMER_H__

#include <chrono>

class ParticleTimer
{
public:
    ParticleTimer(int nParticles);
    void startIteration();
    void endIteration();
    void printParticleEvaluatedPerSecond();

protected: // Data
    int m_numParticles;
    std::chrono::high_resolution_clock::time_point m_startTime;
    std::chrono::high_resolution_clock::time_point m_endTime;
    double m_totalElapsedTime;
    int m_iterationCount;
};

#endif
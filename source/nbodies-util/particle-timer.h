// ==================================
// Created by Juanxpeke on 01-06-2023
// ==================================

#ifndef __PARTICLETIMER_H__
#define __PARTICLETIMER_H__

#include <chrono>
#include <iostream>
#include <map>
#include <vector>

class ParticleTimer
{
public:
    ParticleTimer(int nParticles, const std::string &dataFoldername);
    void startIteration();
    void endIteration();
    void exportData(const std::string &folder);

protected: // Data
    int m_numParticles;
    std::chrono::high_resolution_clock::time_point m_startTime;
    std::chrono::high_resolution_clock::time_point m_endTime;
    double m_totalElapsedTime;
    int m_iterationCount;
    std::map<std::string,std::vector<std::pair<double, double>>> store;  // store['example-timer'][0] = (t0, value)
    std::string m_dataFolderName;
    bool m_dataExported;
};

#endif
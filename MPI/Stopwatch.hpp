#pragma once

#include <mpi.h>

class StopWatch
{
private:
    double startTime;
    double endTime;
public:
    void start();
    void stop();
    void displayTime();
};

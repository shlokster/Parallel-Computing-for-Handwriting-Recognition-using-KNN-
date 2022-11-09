#include "Stopwatch.hpp"

#include <iostream>

void StopWatch::start()
{
    startTime = MPI_Wtime();

}

void StopWatch::stop()
{
    endTime = MPI_Wtime();
}

void StopWatch::displayTime()
{
    double durationInMs = (endTime - startTime) * 1000;
    std::cout << "took " << durationInMs << " ms" << std::endl;
    // std::cout << durationInMs << std::endl;
}

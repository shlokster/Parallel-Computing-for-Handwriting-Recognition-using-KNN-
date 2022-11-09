#include "Stopwatch.hpp"

#include <iostream>

void StopWatch::start()
{
    startTime = std::chrono::high_resolution_clock::now();

}

void StopWatch::stop()
{
    endTime = std::chrono::high_resolution_clock::now();
}

void StopWatch::displayTime()
{
    std::chrono::duration<double, std::milli> duration = endTime - startTime;
    std::cout << "took " << duration.count() << " ms" << std::endl;
    // std::cout << duration.count() << std::endl;
}
// void StopWatch::displayTime()
// {
//     double durationInMs = (endTime - startTime) * 1000;
//     std::cout << "took " << durationInMs << " ms" << std::endl;
//     // std::cout << durationInMs << std::endl;
// }

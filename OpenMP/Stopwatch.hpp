#pragma once

#include <chrono>
#include <ratio>
#include <thread>

class StopWatch
{
public:
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
private:
    TimePoint startTime;
    TimePoint endTime;
public:
    void start();
    void stop();
    void displayTime();
};

#pragma once

class Stopwatch
{
private:
    cudaEvent_t startTime;
    cudaEvent_t stopTime;
    float time;

public:
    Stopwatch();
    void start();
    void stop();
    void displayTime();
};

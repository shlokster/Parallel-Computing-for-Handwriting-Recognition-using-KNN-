#include "Stopwatch.cuh"

#include <iostream>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

Stopwatch::Stopwatch()
{
    HANDLE_ERROR(cudaEventCreate(&startTime));
    HANDLE_ERROR(cudaEventCreate(&stopTime));
}

void Stopwatch::start()
{
    HANDLE_ERROR(cudaEventRecord(startTime, 0));
}

void Stopwatch::stop()
{
    HANDLE_ERROR(cudaEventRecord(stopTime, 0));
    HANDLE_ERROR(cudaEventSynchronize(stopTime));
}

void Stopwatch::displayTime()
{
    HANDLE_ERROR(cudaEventElapsedTime(&time, startTime, stopTime));
    std::cout << "Elapsed time: " << time << " ms" << std::endl;
}

#include "Scalers.cuh"

#include <cmath>
#include <utility>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Stopwatch.cuh"

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

namespace
{
using namespace std;

const uint32_t BLOCK_DIM = 250;

const uint32_t ROWS_AMOUNT = 20000;
const uint32_t ATTRIBUTES_AMOUNT = 16;

namespace NormalizationGPU
{
__device__ void findLocalMinMax(double *devAttributes, double *mins, double *maxes)
{
    int thisThreadStart = threadIdx.x * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
    const int nextThreadStart = (threadIdx.x + 1) * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
    double localMin = devAttributes[thisThreadStart];
    double localMax = localMin;
    __syncthreads();
    for (int row = thisThreadStart; row < nextThreadStart; ++row)
    {
        auto value = devAttributes[row];
        if (value < localMin)
        {
            localMin = value;
        }

        if (value > localMax)
        {
            localMax = value;
        }
    }

    mins[threadIdx.x] = localMin;
    maxes[threadIdx.x] = localMax;
}

__device__ void findMinMax(double *min, double *max, double *localMin, double *localMax)
{
    if (threadIdx.x == 0)
    {
        *min = localMin[0];
        *max = localMax[0];
    }
    __syncthreads();

    for (int i = 0; i < blockDim.x; ++i)
    {
        auto localMinValue = localMin[i];
        if (*min > localMinValue)
        {
            *min = localMinValue;
        }
        auto localMaxValue = localMax[i];
        if (*max < localMaxValue)
        {
            *max = localMaxValue;
        }
    }
}

__device__ void transformValues(double *devAttributes, double *min, double *max)
{
    int thisThreadStart = threadIdx.x * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
    const int nextThreadStart = (threadIdx.x + 1) * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
    double diff = *max - *min;
    for (int row = thisThreadStart; row < nextThreadStart; ++row)
    {
        devAttributes[row] = (devAttributes[row] - *min) / diff;
    }
}

__global__ void normalize(double *devAttributes)
{
    __shared__ double max;
    __shared__ double min;
    {
        __shared__ double localMax[BLOCK_DIM];
        __shared__ double localMin[BLOCK_DIM];
        findLocalMinMax(devAttributes, localMin, localMax);
        __syncthreads();
        findMinMax(&min, &max, localMin, localMax);
        __syncthreads();
    } // scoped shared memory variable localMin and localMax to save memory

    transformValues(devAttributes, &min, &max);
}
} // namespace NormalizationGPU

namespace StandarizationGPU
{
__device__ void findLocalAverage(double *devAttributes, double *averages)
{
    int thisThreadStart = threadIdx.x * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
    const int nextThreadStart = (threadIdx.x + 1) * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
    double localAverage = 0;
    for (int row = thisThreadStart; row < nextThreadStart; ++row)
    {
        localAverage += devAttributes[row];
    }

    averages[threadIdx.x] = localAverage / (nextThreadStart - thisThreadStart);
}

__device__ void findAverage(double *average, double *localAverage)
{
    if (threadIdx.x == 0)
    {
        *average = 0;
    }
    __syncthreads();
    atomicAdd(average, localAverage[threadIdx.x]);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        *average /= blockDim.x;
    }
}

__device__ void findLocalVariation(double *devAttributes, double *variations, double *average)
{
    int thisThreadStart = threadIdx.x * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
    const int nextThreadStart = (threadIdx.x + 1) * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
    double localVariation = 0;
    for (int row = thisThreadStart; row < nextThreadStart; ++row)
    {
        auto tmp = devAttributes[row] - *average;
        localVariation += tmp * tmp;
    }

    variations[threadIdx.x] = localVariation;
}

__device__ void findVariation(double *variation, double *localVariations)
{
    if (threadIdx.x == 0)
    {
        *variation = 0;
    }
    __syncthreads();
    atomicAdd(variation, localVariations[threadIdx.x]);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        *variation /= ROWS_AMOUNT;
        *variation = sqrt(*variation);
    }
}

__device__ void transformValues(double *devAttributes, double *average, double *variation)
{
    int thisThreadStart = threadIdx.x * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
    const int nextThreadStart = (threadIdx.x + 1) * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
    for (int row = thisThreadStart; row < nextThreadStart; ++row)
    {
        devAttributes[row] = (devAttributes[row] - *average) / *variation;
    }
}

__global__ void standarize(double *devAttributes)
{
    __shared__ double average;
    {
        __shared__ double localAverage[BLOCK_DIM];
        findLocalAverage(devAttributes, localAverage);
        __syncthreads();
        findAverage(&average, localAverage);
        __syncthreads();
    } // scoped shared memory variable localAverage to save memory

    __shared__ double variation;
    {
        __shared__ double localVariation[BLOCK_DIM];
        findLocalVariation(devAttributes, localVariation, &average);
        __syncthreads();
        findVariation(&variation, localVariation);
        __syncthreads();
    } // scoped shared memory variable localVariation to save memory

    transformValues(devAttributes, &average, &variation);
}
} // namespace StandarizationGPU
} // namespace

void Scalers::normalize(vector<double> &attributesValues)
{
    double *attributes = attributesValues.data();
    double *devAttributes = nullptr;
    HANDLE_ERROR(cudaMalloc(&devAttributes, attributesValues.size() * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(devAttributes, attributes, attributesValues.size() * sizeof(double), cudaMemcpyHostToDevice));
    NormalizationGPU::normalize<<<ATTRIBUTES_AMOUNT, BLOCK_DIM, (2 + BLOCK_DIM * 2) * sizeof(double)>>>(devAttributes);
    HANDLE_ERROR(cudaMemcpy(attributes, devAttributes, attributesValues.size() * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(devAttributes));
}

void Scalers::standarize(vector<double> &attributesValues)
{
    double *attributes = attributesValues.data();
    double *devAttributes = nullptr;
    HANDLE_ERROR(cudaMalloc(&devAttributes, attributesValues.size() * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(devAttributes, attributes, attributesValues.size() * sizeof(double), cudaMemcpyHostToDevice));
    StandarizationGPU::standarize<<<ATTRIBUTES_AMOUNT, BLOCK_DIM, (2 + BLOCK_DIM) * sizeof(double)>>>(devAttributes);
    HANDLE_ERROR(cudaMemcpy(attributes, devAttributes, attributesValues.size() * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(devAttributes));
}

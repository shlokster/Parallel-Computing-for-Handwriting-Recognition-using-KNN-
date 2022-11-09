#include "LetterRecognition.cuh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

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
using namespace cooperative_groups;

const uint32_t BLOCK_DIM = 50;

const uint32_t ROWS_AMOUNT = 20000;
const uint32_t ATTRIBUTES_AMOUNT = 16;

const uint32_t TRAIN_SET_SIZE = ROWS_AMOUNT * 0.9;
const uint32_t TEST_SET_SIZE = ROWS_AMOUNT - TRAIN_SET_SIZE;

namespace GPU
{
__device__ void calculateSquares(int usedThreads, double *dataset, int testRowIndex)
{
    int thisThreadStart = threadIdx.x * TRAIN_SET_SIZE / usedThreads + blockIdx.x * ROWS_AMOUNT;
    const int nextThreadStart = (threadIdx.x + 1) * TRAIN_SET_SIZE / usedThreads + blockIdx.x * ROWS_AMOUNT;

    // Calculate squares for every attribute
    double testAttribute = dataset[blockIdx.x * ROWS_AMOUNT + testRowIndex];
    for (uint32_t k = thisThreadStart; k < nextThreadStart; ++k)
    {
        double tmp = testAttribute - dataset[k];
        dataset[k] = tmp * tmp;
    }
}

__device__ void calculateSums(int usedThreads, double *dataset)
{
    int firstAttributeIndex = threadIdx.x * TRAIN_SET_SIZE / usedThreads;

    int thisThreadStart = threadIdx.x * TRAIN_SET_SIZE / usedThreads + blockIdx.x * ROWS_AMOUNT;
    const int nextThreadStart = (threadIdx.x + 1) * TRAIN_SET_SIZE / usedThreads + blockIdx.x * ROWS_AMOUNT;

    // Sum each row & calculate square root
    for (uint32_t k = thisThreadStart; k < nextThreadStart; ++k)
    {
        atomicAdd(&(dataset[firstAttributeIndex++]), dataset[k]);
    }
}

__device__ void calculateSquaredRoots(int usedThreads, double *dataset)
{
    // Split only first row (train data) into blocks; then splitted rows into block split into threads
    int thisThreadStart = threadIdx.x * TRAIN_SET_SIZE / gridDim.x / usedThreads + blockIdx.x * TRAIN_SET_SIZE / gridDim.x;
    int nextThreadStart = (threadIdx.x + 1) * TRAIN_SET_SIZE / gridDim.x / usedThreads + blockIdx.x * TRAIN_SET_SIZE / gridDim.x;

    // Sum each row & calculate square root
    for (uint32_t k = thisThreadStart; k < nextThreadStart; ++k)
    {
        dataset[k] = sqrt(dataset[k]);
    }
}

__device__ void findLocalMinSumWithIndex(int usedThreads, double *dataset, double *mins, int *indices)
{
    // Split only first row (train data) into blocks; then splitted rows into block split into threads
    int thisThreadStart = threadIdx.x * TRAIN_SET_SIZE / gridDim.x / usedThreads + blockIdx.x * TRAIN_SET_SIZE / gridDim.x;
    const int nextThreadStart = (threadIdx.x + 1) * TRAIN_SET_SIZE / gridDim.x / usedThreads + blockIdx.x * TRAIN_SET_SIZE / gridDim.x;
    double localMin = dataset[thisThreadStart];
    int localMinIndex = thisThreadStart;
    __syncthreads();
    for (int row = thisThreadStart; row < nextThreadStart; ++row)
    {
        auto value = dataset[row];
        if (value < localMin)
        {
            localMin = value;
            localMinIndex = row;
        }
    }

    mins[threadIdx.x] = localMin;
    indices[threadIdx.x] = localMinIndex;
}

__device__ void findMinSumWithIndex(int usedThreads, double *min, int *minIndex, double *localMin, int *localMinIndices)
{
    for (int i = 0; i < usedThreads; ++i)
    {
        auto localMinValue = localMin[i];
        if (*min > localMinValue)
        {
            *min = localMinValue;
            *minIndex = localMinIndices[i];
        }
    }
}

__global__ void knn(double *dataset, int testRowIndex, double *min, int *minIndex)
{
    // limit threads to get equal distribution of data across all threads
    int usedThreads = blockDim.x;
    while (TRAIN_SET_SIZE % usedThreads != 0)
    {
        --usedThreads;
    }
    if (threadIdx.x < usedThreads)
    {
        calculateSquares(usedThreads, dataset, testRowIndex);
        __syncthreads();
        // skip first column as others columns will add values into first column
        if (blockIdx.x != 0)
        {
            calculateSums(usedThreads, dataset);
            __syncthreads();
        }
    }
    grid_group grid = this_grid();
    grid.sync();

    // limit threads to get equal distribution of data across all threads
    usedThreads = blockDim.x;
    while ((TRAIN_SET_SIZE / gridDim.x) % usedThreads != 0)
    {
        --usedThreads;
    }
    if (threadIdx.x < usedThreads)
    {
        calculateSquaredRoots(usedThreads, dataset);
        __syncthreads();

        __shared__ double localMin[BLOCK_DIM];
        __shared__ int localMinIndices[BLOCK_DIM];
        findLocalMinSumWithIndex(usedThreads, dataset, localMin, localMinIndices);
        __syncthreads();
        findMinSumWithIndex(usedThreads, min, minIndex, localMin, localMinIndices);
        __syncthreads();
    }
}
} // namespace GPU
} // namespace

auto LetterRecognition::fetchData(const string &path) -> LetterData
{
    const uint32_t MATRIX_SIZE = ATTRIBUTES_AMOUNT + 1; // attributes + its class

    LetterData data;
    vector<vector<double>> matrix;
    matrix.resize(ATTRIBUTES_AMOUNT);
    for (auto &attributes : matrix)
    {
        attributes.reserve(ROWS_AMOUNT);
    }
    data.letters.reserve(ROWS_AMOUNT);
    data.attributesAmount = ATTRIBUTES_AMOUNT;

    ifstream file(path.c_str());
    string line;

    while (getline(file, line))
    {
        stringstream stream(line);
        string stringValue; // represents double value
        uint32_t position = 0;
        while (getline(stream, stringValue, ','))
        {
            if (position == 0)
            {
                data.letters.push_back(stringValue.at(0));
            }
            else
            {
                matrix.at(position - 1).push_back(stod(stringValue));
            }
            position = (position + 1) % MATRIX_SIZE;
        }
    }

    // Flatten 2D array
    data.attributes.reserve(ATTRIBUTES_AMOUNT * ROWS_AMOUNT);

    for (auto &column : matrix)
    {
        for (auto &value : column)
        {
            data.attributes.push_back(value);
        }
    }

    return data;
}

void LetterRecognition::Result::printOverallResult()
{
    double percentage = static_cast<double>(correct) / static_cast<double>(all) * 100.0;
    std::cout << "Accuracy: " << correct << "/" << all
              << ", Percentage: " << percentage << "%" << std::endl;
}

auto LetterRecognition::knn(LetterData &letterData) -> Result
{
    double *devAttributes;
    HANDLE_ERROR(cudaMalloc(&devAttributes, ATTRIBUTES_AMOUNT * ROWS_AMOUNT * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(devAttributes, letterData.attributes.data(), ATTRIBUTES_AMOUNT * ROWS_AMOUNT * sizeof(double), cudaMemcpyHostToDevice));
    // Copy dataset for each test row
    Result result{0, 0};
    for (int i = TRAIN_SET_SIZE; i < TRAIN_SET_SIZE + TEST_SET_SIZE; ++i)
    {
        double *dataset = nullptr;
        HANDLE_ERROR(cudaMalloc(&dataset, ATTRIBUTES_AMOUNT * ROWS_AMOUNT * sizeof(double)));
        cudaDeviceSynchronize();
        HANDLE_ERROR(cudaMemcpy(dataset, devAttributes, ATTRIBUTES_AMOUNT * ROWS_AMOUNT * sizeof(double), cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize();

        double *devMinValue;
        HANDLE_ERROR(cudaMalloc(&devMinValue, sizeof(double)));
        HANDLE_ERROR(cudaMemset(devMinValue, 1000000.0, sizeof(double))); // randomly high value
        int *devMinIndex;
        HANDLE_ERROR(cudaMalloc(&devMinIndex, sizeof(int)));

        dim3 dimGrid(ATTRIBUTES_AMOUNT, 1, 1);
        dim3 dimBlock(BLOCK_DIM, 1, 1);
        void *kernelArgs[] = {
            (void *)&dataset,
            (void *)&i,
            (void *)&devMinValue,
            (void *)&devMinIndex,
        };
        int sharedMemorySize = BLOCK_DIM * (sizeof(double) + sizeof(int));
        HANDLE_ERROR(cudaLaunchCooperativeKernel((void *)GPU::knn, dimGrid, dimBlock, kernelArgs, sharedMemorySize, nullptr));
        HANDLE_ERROR(cudaFree(devMinValue));
        int minIndex{};
        HANDLE_ERROR(cudaMemcpy(&minIndex, devMinIndex, sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaFree(devMinIndex));

        char predictedGenre = letterData.letters[minIndex];
        auto actualGenre = letterData.letters[i];
        if (predictedGenre == actualGenre)
            result.correct++;

        HANDLE_ERROR(cudaFree(dataset));
    }

    result.all = TEST_SET_SIZE;

    HANDLE_ERROR(cudaFree(devAttributes));

    return result;
}

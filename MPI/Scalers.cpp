#include "Scalers.hpp"

#include <cmath>
#include <iostream>
#include <mpi.h>

namespace
{
    using namespace std;
}

void Scalers::normalizeMPI(vector<vector<double>>* attributeSet, int index)
{
    double minMax[2] {0.,};
    if (mpiWrapper.getWorldRank() == 0)
    {
        const auto [min, max] = findMinMax(attributeSet->at(index));
        minMax[0] = min;
        minMax[1] = max;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (int errorCode = MPI_Bcast(minMax, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        errorCode != MPI_SUCCESS)
    {
        cout << "Failed to broadcast data! Error:" << errorCode << ". Rank: " << mpiWrapper.getWorldRank() << endl;
    }

    double* set;
    if (mpiWrapper.getWorldRank() == 0)
    {
        set = attributeSet->at(index).data();
    }
    vector<double> subset(ROWS_AMOUNT / mpiWrapper.getWorldSize() + 1);

    auto [sendCounts, displacements] = mpiWrapper.calculateDisplacements(ROWS_AMOUNT);
    MPI_Scatterv(set, sendCounts.data(), displacements.data(), MPI_DOUBLE, subset.data(), sendCounts.at(mpiWrapper.getWorldRank()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double diff = minMax[1] - minMax[0];

    for (auto& value : subset)
    {
        value = (value - minMax[0]) / diff;
    }

    MPI_Gatherv(subset.data(), sendCounts.at(mpiWrapper.getWorldRank()), MPI_DOUBLE, set, sendCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Scalers::normalize(vector<double> &attributeSet)
{
    const auto [min, max] = findMinMax(attributeSet);

    double diff = max - min;

    for (auto& value : attributeSet)
    {
        value = (value - min) / diff;
    }
}

pair<double, double> Scalers::findMinMax(vector<double> &attributeSet)
{
    double min = attributeSet.at(0);
    double max = min;

    for (const auto& value : attributeSet)
    {
        if (value < min)
        {
            min = value;
        }
        
        if (value > max)
        {
            max = value;
        }
    }

    return std::make_pair(min, max);
}

void Scalers::standarizeMPI(vector<vector<double>>* attributeSet, int index)
{
    double averageVariation[2] {0.,};
    if (mpiWrapper.getWorldRank() == 0)
    {
        const auto [average, variation] = findAverageAndVariation(attributeSet->at(index));
        averageVariation[0] = average;
        averageVariation[1] = variation;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (int errorCode = MPI_Bcast(averageVariation, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        errorCode != MPI_SUCCESS)
    {
        cout << "Failed to broadcast data! Error:" << errorCode << ". Rank: " << mpiWrapper.getWorldRank() << endl;
    }

    double* set;
    if (mpiWrapper.getWorldRank() == 0)
    {
        set = attributeSet->at(index).data();
    }
    vector<double> subset(ROWS_AMOUNT / mpiWrapper.getWorldSize() + 1);

    auto [sendCounts, displacements] = mpiWrapper.calculateDisplacements(ROWS_AMOUNT);
    MPI_Scatterv(set, sendCounts.data(), displacements.data(), MPI_DOUBLE, subset.data(), sendCounts.at(mpiWrapper.getWorldRank()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (auto& value : subset)
    {
        value = (value - averageVariation[0]) / averageVariation[1];
    }

    MPI_Gatherv(subset.data(), sendCounts.at(mpiWrapper.getWorldRank()), MPI_DOUBLE, set, sendCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Scalers::standarize(vector<double> &attributeSet)
{
    const auto [average, variation] = findAverageAndVariation(attributeSet);

    for (auto& value : attributeSet)
    {
        value = (value - average) / variation;
    }
}

pair<double, double> Scalers::findAverageAndVariation(vector<double> &attributeSet)
{
    double average{};
    
    for (const auto& value : attributeSet)
    {
        average += value;
    }
    average /= attributeSet.size();

    double variation{};
    for (const auto& value : attributeSet)
    {
        auto tmp = value - average;
        variation += tmp * tmp;
    }
    variation /= attributeSet.size(); // variance
    variation = sqrt(variation);

    return std::make_pair(average, variation);
}

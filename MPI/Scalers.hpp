#pragma once

#include <vector>
#include <utility>

#include "MpiWrapper.hpp"

namespace
{
    using namespace std;
}

class Scalers
{
public:
    Scalers(MpiWrapper& mpi) : ROWS_AMOUNT{20000}, mpiWrapper{mpi} {}
    void normalizeMPI(vector<vector<double>>* attributeSet, int index);
    void normalize(vector<double> &attributeSet);
    void standarizeMPI(vector<vector<double>>* attributeSet, int index);
    void standarize(vector<double> &attributeSet);
private:
    const int ROWS_AMOUNT;
    MpiWrapper& mpiWrapper;

    pair<double, double> findMinMax(vector<double> &attributeSet);
    pair<double, double> findAverageAndVariation(vector<double> &attributeSet);
};

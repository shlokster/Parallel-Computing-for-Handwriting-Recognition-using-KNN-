#pragma once

#include <vector>
#include <utility>

namespace
{
    using namespace std;
}

class Scalers
{
    pair<double, double> findMinMax(vector<double> &attributeSet);
    pair<double, double> findAverageAndVariation(vector<double> &attributeSet);
public:
    void normalize(vector<double> &attributeSet);
    void standarize(vector<double> &attributeSet);
};

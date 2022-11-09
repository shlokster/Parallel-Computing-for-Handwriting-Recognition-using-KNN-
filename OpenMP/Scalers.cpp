#include "Scalers.hpp"

#include <cmath>

namespace
{
    using namespace std;
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

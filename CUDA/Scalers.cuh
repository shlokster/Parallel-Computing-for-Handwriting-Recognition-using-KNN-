#pragma once

#include <vector>

namespace
{
using namespace std;
}

struct Scalers
{
    void normalize(vector<double> &attributeSet);
    void standarize(vector<double> &attributeSet);
};

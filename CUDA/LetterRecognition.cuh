#pragma once

#include <vector>
#include <map>
#include <set>
#include <string>
#include <cstdint>

namespace
{
using namespace std;
}

struct LetterRecognition
{
    struct LetterData
    {
        vector<double> attributes;
        vector<char> letters;
        uint32_t attributesAmount;
    };

    struct Result
    {
        uint32_t correct;
        uint32_t all;

        void printOverallResult();
    };

    LetterData fetchData(const string &path);
    Result knn(LetterData &letterData);
};

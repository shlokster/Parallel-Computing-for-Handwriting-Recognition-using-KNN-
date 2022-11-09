#pragma once

#include <vector>
#include <map>
#include <set>
#include <string>
#include <cstdint>

#include "MpiWrapper.hpp"

namespace
{
using namespace std;
}

class LetterRecognition
{
public:
    struct LetterData
    {
        vector<vector<double>> attributes{};
        vector<char> letters{};
        uint32_t attributesAmount{16};
    };

    struct Result
    {
        uint32_t correct{};
        uint32_t all{};
        map<char, std::pair<uint32_t, uint32_t>> confusionMatrix{}; // first pair value = correctly recognized values; second pair value = incorrectly recognized values

        void printOverallResult();
        void printConfustionMatrix();
    };

    LetterRecognition(MpiWrapper &mpi) : mpiWrapper{mpi} {}
    LetterData fetchData(const string &path);
    Result knnMPI(LetterData &letterData);
    Result knnMPI(LetterData &letterData, uint32_t neighbours);
    void crossValidationMPI(LetterData &letterData, uint32_t neighbours);

private:
    uint32_t SET_SIZE = 20000;
    uint32_t ATTRIBUTES = 16;
    uint32_t MATRIX_SIZE = ATTRIBUTES + 1; // attributes + its class

    MpiWrapper &mpiWrapper;

    char voteOnGenre(const set<pair<double, char>> &nearestNeighbours);
    void broadcastLetterData(LetterData &letterData);
    Result runKnn(LetterData &trainData, LetterData &testData);
    Result runKnn(LetterData &trainData, LetterData &testData, uint32_t neighbours);
};

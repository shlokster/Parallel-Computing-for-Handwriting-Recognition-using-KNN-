#include "LetterRecognition.hpp"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <set>
#include <mpi.h>

namespace
{
using namespace std;
}

auto LetterRecognition::fetchData(const string &path) -> LetterData
{
    LetterData data;
    data.attributes.resize(ATTRIBUTES);
    for (auto &attributes : data.attributes)
    {
        attributes.reserve(SET_SIZE);
    }
    data.letters.reserve(SET_SIZE);
    data.attributesAmount = ATTRIBUTES;

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
                data.attributes.at(position - 1).push_back(stod(stringValue));
            }
            position = (position + 1) % MATRIX_SIZE;
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

void LetterRecognition::Result::printConfustionMatrix()
{
    for (const auto &entry : confusionMatrix)
    {
        double percentage = static_cast<double>(entry.second.first) / static_cast<double>(entry.second.first + entry.second.second) * 100.0;
        std::cout << "Letter: " << entry.first << ",\tpercentage: " << percentage << "%,\tcorrect: " << entry.second.first << ",\tincorrect: " << entry.second.second << std::endl;
    }
}

auto LetterRecognition::knnMPI(LetterData &letterData) -> Result
{
    return knnMPI(letterData, 1);
}

auto LetterRecognition::knnMPI(LetterData &letterData, uint32_t neighbours) -> Result
{
    const uint32_t TRAIN_SET_SIZE = SET_SIZE * 0.9;

    // Split letterData into train and test data
    vector<vector<double>> testAttributeData(ATTRIBUTES);
    vector<char> testClassData;
    if (mpiWrapper.getWorldRank() == 0)
    {
        for (auto &data : testAttributeData)
        {
            data.reserve(SET_SIZE - TRAIN_SET_SIZE);
        }
        testClassData.reserve(SET_SIZE - TRAIN_SET_SIZE);

        for (uint32_t i = 0; i < ATTRIBUTES; ++i)
        {
            auto &wholeData = letterData.attributes.at(i);
            auto &testData = testAttributeData.at(i);
            testData.insert(testData.end(), make_move_iterator(wholeData.begin() + TRAIN_SET_SIZE), make_move_iterator(wholeData.end()));
            wholeData.resize(TRAIN_SET_SIZE);
        }

        testClassData.insert(testClassData.end(), make_move_iterator(letterData.letters.begin() + TRAIN_SET_SIZE), make_move_iterator(letterData.letters.end()));
        letterData.letters.resize(TRAIN_SET_SIZE);
    }

    // Broadcast train data - letters (first adjust array size)
    if (letterData.letters.size() < TRAIN_SET_SIZE)
    {
        letterData.letters.resize(TRAIN_SET_SIZE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (int errorCode = MPI_Bcast(letterData.letters.data(), TRAIN_SET_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
        errorCode != MPI_SUCCESS)
    {
        cout << "Failed to broadcast data! Error:" << errorCode << ". Rank: " << mpiWrapper.getWorldRank() << endl;
    }

    // Broadcast train data - attributes (first adjust array size)
    if (letterData.attributes.size() != ATTRIBUTES)
    {
        letterData.attributes.resize(ATTRIBUTES);
        for (auto &set : letterData.attributes)
        {
            set.resize(TRAIN_SET_SIZE);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (uint32_t i = 0; i < ATTRIBUTES; ++i)
    {
        if (int errorCode = MPI_Bcast(letterData.attributes.at(i).data(), TRAIN_SET_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            errorCode != MPI_SUCCESS)
        {
            cout << "Failed to broadcast data! Error:" << errorCode << ". Rank: " << mpiWrapper.getWorldRank() << endl;
        }
    }

    // Scatter test data
    auto [sendCounts, displacements] = mpiWrapper.calculateDisplacements(0.1 * SET_SIZE);

    vector<vector<double>> testAttributeDataSubset(ATTRIBUTES);
    vector<char> testClassDataSubset(sendCounts.at(mpiWrapper.getWorldRank()));

    for (uint32_t i = 0; i < ATTRIBUTES; ++i)
    {
        testAttributeDataSubset.at(i).resize(sendCounts.at(mpiWrapper.getWorldRank()));
        MPI_Scatterv(testAttributeData.at(i).data(), sendCounts.data(), displacements.data(), MPI_DOUBLE, testAttributeDataSubset.at(i).data(), sendCounts.at(mpiWrapper.getWorldRank()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Scatterv(testClassData.data(), sendCounts.data(), displacements.data(), MPI_CHAR, testClassDataSubset.data(), sendCounts.at(mpiWrapper.getWorldRank()), MPI_CHAR, 0, MPI_COMM_WORLD);

    LetterData testData;
    testData.attributes = testAttributeDataSubset;
    testData.letters = testClassDataSubset;
    MPI_Barrier(MPI_COMM_WORLD);
    // Run KNN
    Result result;
    if (neighbours == 1)
    {
        result = runKnn(letterData, testData);
    } else {
        result = runKnn(letterData, testData, neighbours);
    }
    uint32_t correctAllSubResult[2]{result.correct, result.all};
    // Gather results
    vector<uint32_t> correctAll(2 * mpiWrapper.getWorldSize());

    MPI_Barrier(MPI_COMM_WORLD);
    //   MPI_Allgather(correctAllSubResult, 2, MPI_UNSIGNED, correctAll.data(), 2, MPI_UNSIGNED, MPI_COMM_WORLD);
    MPI_Gather(correctAllSubResult, 2, MPI_UNSIGNED, correctAll.data(), 2, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Merge results into one result object
    Result gatheredResult;
    for (uint32_t i = 0; i < correctAll.size(); ++i)
    {
        if (i % 2 == 0)
        {
            gatheredResult.correct += correctAll.at(i);
        }
        else
        {
            gatheredResult.all += correctAll.at(i);
        }
    }

    return gatheredResult;
}

auto LetterRecognition::runKnn(LetterData &trainData, LetterData &testData) -> Result
{
    uint32_t i;
    vector<vector<double>> dataset;
    Result result{0, 0, {}};
    for (i = 0; i < testData.letters.size(); ++i)
    {
        dataset = trainData.attributes;

        // Calculate squares for every attribute
        uint32_t j;
        for (j = 0; j < trainData.attributesAmount; ++j)
        {
            double testAttribute = testData.attributes.at(j).at(i);
            uint32_t k;
            for (k = 0; k < dataset.at(0).size(); ++k)
            {
                double tmp = testAttribute - dataset.at(j).at(k);
                dataset.at(j).at(k) = tmp * tmp;
            }
        }

        uint32_t k;
        double minimalSum;
        char predictedGenre = '0';
        // Sum each row & calculate square root
        for (k = 0; k < trainData.letters.size(); ++k)
        {
            double sum = 0.0;
            uint32_t a;
            for (a = 0; a < ATTRIBUTES; ++a)
            {
                sum += dataset.at(a).at(k);
            }

            sum = sqrt(sum);

            if (k == 0 || sum < minimalSum)
            {
                minimalSum = sum;
                predictedGenre = trainData.letters.at(k);
            }
        }

        auto actualGenre = testData.letters.at(i);
        // Add result to overall result & confusion result
        if (result.confusionMatrix.find(actualGenre) == result.confusionMatrix.end())
        {
            result.confusionMatrix[actualGenre] = make_pair(0, 0);
        }

        if (predictedGenre == actualGenre)
        {
            ++result.confusionMatrix[actualGenre].first;
            ++result.correct;
        }
        else
        {
            ++result.confusionMatrix[actualGenre].second;
        }
    }

    result.all = testData.letters.size();

    return result;
}

auto LetterRecognition::runKnn(LetterData &trainData, LetterData &testData, uint32_t neighbours) -> Result
{
    uint32_t i;
    vector<vector<double>> dataset;
    Result result{0, 0, {}};
    for (i = 0; i < testData.letters.size(); ++i)
    {
        dataset = trainData.attributes;

        // Calculate squares for every attribute
        uint32_t j;
        for (j = 0; j < trainData.attributesAmount; ++j)
        {
            double testAttribute = testData.attributes.at(j).at(i);
            uint32_t k;
            for (k = 0; k < dataset.at(0).size(); ++k)
            {
                double tmp = testAttribute - dataset.at(j).at(k);
                dataset.at(j).at(k) = tmp * tmp;
            }
        }

        set<pair<double, char>> nearestNeighbours;
        uint32_t k;
        // Sum each row & calculate square root
        for (k = 0; k < trainData.letters.size(); ++k)
        {
            double sum = 0.0;
            char genre = '0';
            uint32_t a;
            for (a = 0; a < ATTRIBUTES; ++a)
            {
                sum += dataset.at(a).at(k);
            }

            sum = sqrt(sum);
            genre = trainData.letters.at(k);

            if (k < neighbours)
            {
                nearestNeighbours.emplace(make_pair(sum, genre));
            }
            else if ((*--nearestNeighbours.end()).first > sum)
            {
                nearestNeighbours.erase(--nearestNeighbours.end());
                nearestNeighbours.emplace(make_pair(sum, genre));
            }
        }

        // Vote/decide which neighbour
        auto predictedGenre = voteOnGenre(nearestNeighbours);
        auto actualGenre = testData.letters.at(i);
        // Add result to overall result & confusion result
        if (result.confusionMatrix.find(actualGenre) == result.confusionMatrix.end())
        {
            result.confusionMatrix[actualGenre] = make_pair(0, 0);
        }

        if (predictedGenre == actualGenre)
        {
            ++result.confusionMatrix[actualGenre].first;
            ++result.correct;
        }
        else
        {
            ++result.confusionMatrix[actualGenre].second;
        }
    }

    result.all = testData.letters.size();

    return result;
}

char LetterRecognition::voteOnGenre(const set<pair<double, char>> &nearestNeighbours)
{
    map<char, uint32_t> occurencesMap;

    for (const auto &entry : nearestNeighbours)
    {
        char neighbour = entry.second;
        if (occurencesMap.find(neighbour) == occurencesMap.end())
        {
            occurencesMap[neighbour] = 1;
        }
        else
        {
            ++occurencesMap[neighbour];
        }
    }

    char chosenChar;
    uint32_t chosenCharOccurences = 0;
    for (const auto &entry : occurencesMap)
    {
        if (entry.second > chosenCharOccurences)
        {
            chosenChar = entry.first;
            chosenCharOccurences = entry.second;
        }
    }

    return chosenChar;
}

void LetterRecognition::crossValidationMPI(LetterData &letterData, uint32_t neighbours)
{
    const int ITERATIONS = 10;
    cout << "Cross validating for " << ITERATIONS << " subsets..." << endl;
    uint32_t correct = 0;
    uint32_t all = 0;
    for (int i = 0; i < ITERATIONS; ++i)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpiWrapper.getWorldRank() == 0)
        {
            for (auto &attributeSet : letterData.attributes)
            {
                std::rotate(attributeSet.begin(), attributeSet.begin() + (SET_SIZE * 0.1), attributeSet.end());
            }
            std::rotate(letterData.letters.begin(), letterData.letters.begin() + (SET_SIZE * 0.1), letterData.letters.end());
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto letterDataCopy = letterData; // Hack, because of letterData is modified by KNN 
        auto result = knnMPI(letterData, neighbours);
        letterData = letterDataCopy;      // Hack, because of letterData is modified by KNN 
       
        if (mpiWrapper.getWorldRank() == 0)
        {
            correct += result.correct;
            all += result.all;
            cout << "Subset " << i + 1 << " results: " << endl;
            result.printOverallResult();
        }
    }

    if (mpiWrapper.getWorldRank() == 0)
    {
        cout << "Overall cross validation results: " << endl;
        Result result{correct, all, {}};
        result.printOverallResult();
    }
}

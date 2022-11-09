#include <cstdint>
#include <iostream>

#include "MpiWrapper.hpp"
#include "Stopwatch.hpp"
#include "LetterRecognition.hpp"
#include "Scalers.hpp"

namespace
{
using namespace std;
}

int main()
{
    MpiWrapper mpiWrapper;
    LetterRecognition letterRecognition(mpiWrapper);
    Scalers scalers(mpiWrapper);
    StopWatch timer;

    LetterRecognition::LetterData letterData;
    if (mpiWrapper.getWorldRank() == 0)
    {
        const string DATASET_PATH{"../csv/letter-recognition.csv"};
        letterData = letterRecognition.fetchData(DATASET_PATH);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (unsigned int i = 0; i < letterData.attributesAmount; ++i)
    {
        // scalers.normalizeMPI(&letterData.attributes, i);
        scalers.standarizeMPI(&letterData.attributes, i);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    timer.start();
    auto results = letterRecognition.knnMPI(letterData, 5);
    MPI_Barrier(MPI_COMM_WORLD);
    timer.stop();
    if (mpiWrapper.getWorldRank() == 0)
    {
        timer.displayTime();
        results.printOverallResult();
    }

    // // // Beware of runing knnMPI before this method - knnMPI modifies letterData!
    // timer.start();
    // letterRecognition.crossValidationMPI(letterData, 1);
    // timer.stop();
    // timer.displayTime();

    return 0;
}

#include <iostream>

#include "LetterRecognition.cuh"
#include "Scalers.cuh"
#include "Stopwatch.cuh"

namespace
{
using namespace std;
}

int main()
{
    LetterRecognition letterRecognition;
    Scalers scalers;
    Stopwatch timer;
    const string DATASET_PATH{"../csv/letter-recognition.csv"};

    auto letterData = letterRecognition.fetchData(DATASET_PATH);

    timer.start();
    // scalers.normalize(letterData.attributes);
    scalers.standarize(letterData.attributes);
    timer.stop();
    // cout << "Normalization: ";
    cout << "Standarization: ";
    timer.displayTime();

    timer.start();
    auto results = letterRecognition.knn(letterData);
    timer.stop();
    cout << "KNN: ";
    timer.displayTime();

    results.printOverallResult();

    return 0;
}

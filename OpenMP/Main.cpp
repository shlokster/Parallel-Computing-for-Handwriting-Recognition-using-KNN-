#include <cstdint>
#include <iostream>
#include <omp.h>

#include "Stopwatch.hpp"
#include "LetterRecognition.hpp"
#include "Scalers.hpp"

namespace
{
    using namespace std;
}

int main()
{
    LetterRecognition letterRecognition;
    Scalers scalers;
    StopWatch timer;
    const string DATASET_PATH{"../csv/letter-recognition.csv"};

    auto letterData = letterRecognition.fetchData(DATASET_PATH);

    uint32_t i;
    // #pragma omp parallel for shared(letterData) private(i) num_threads(2)
    #pragma omp parallel for shared(letterData) private(i)
    for (i = 0; i < letterData.attributesAmount; ++i)
    {
        // scalers.normalize(letterData.attributes.at(i));
        scalers.standarize(letterData.attributes.at(i));
    }


    for (int i=0; i<16; i++) {
        cout<<"Standarized: "<<letterData.attributes[0][i]<<endl;
    }
    timer.start();

    // letterRecognition.crossValidation(letterData, 5);
    auto results = letterRecognition.knn(letterData);
    // auto results = letterRecognition.knn(letterData, 5);
    timer.stop();
    timer.displayTime();
    
    // results.printConfustionMatrix();
    results.printOverallResult();

    return 0;
}

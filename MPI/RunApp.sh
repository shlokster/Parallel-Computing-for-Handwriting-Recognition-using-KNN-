#!/bin/bash

echo "Cleaning"
rm app
echo "Compiling app"
mpic++ -std=gnu++17 -Wall -fopenmp Main.cpp Stopwatch.cpp LetterRecognition.cpp Scalers.cpp MpiWrapper.cpp -o app
echo "Running app"
mpiexec -np 4 ./app

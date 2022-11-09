#!/bin/bash

echo "Cleaning"
rm app
echo "Compiling app"
g++ -std=gnu++17 -Wall -fopenmp Main.cpp Stopwatch.cpp LetterRecognition.cpp Scalers.cpp -o app
echo "Running app"
./app

#echo "Cleaning"
rm app
#echo "Compiling app"
nvcc  -arch=sm_60 -Xcompiler -std=gnu++14 -rdc=true -o app Main.cu Scalers.cu LetterRecognition.cu Stopwatch.cu
echo "Running app"
./app

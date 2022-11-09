from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array
import numpy as np
import datetime
import csv

class LetterData:
    dataset = []
    genres = []

def fetchData(path):
    data = LetterData()
    with open(path, newline='') as csvfile:
        fileData = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in fileData:
            data.dataset.append(row[1:])
            data.genres.append(row[:1][0])
    # fdata = np.array(data, dtype=float) #  convert using numpy
    # fdata = [float(i) for i in a] #  convert with for loop
    # check_array(data, dtype='numeric')
    return data

def knn(letterData):
    # knn = KNeighborsClassifier(n_neighbors=1)
    knn = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree', n_jobs = 2)
    trainingSetSize = int(len(letterData.dataset) * 0.9)

    knn.fit(letterData.dataset[:trainingSetSize], letterData.genres[:trainingSetSize])
    prediction = knn.predict(letterData.dataset[trainingSetSize:])

    accuracy = accuracy_score(prediction, letterData.genres[trainingSetSize:])
    print('Accuracy: ', accuracy*100, '%')

def main():
    data = fetchData('csv/letter-recognition.csv')

    # Normalization
    normalizedDataset = LetterData()
    normalizedDataset.dataset = MinMaxScaler().fit_transform(data.dataset)
    normalizedDataset.genres = data.genres

    #Standarization
    standarizedDataset = LetterData()
    standarizedDataset.dataset = StandardScaler().fit_transform(data.dataset)
    standarizedDataset.genres = data.genres

    start = datetime.datetime.now()
    knn(data)
    end = datetime.datetime.now()
    knn(normalizedDataset)
    knn(standarizedDataset)

    elapsed = end - start
    print('Time: ', elapsed)

if __name__ == "__main__":
    main()
import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import DataReader


class MeanShift:
    def __init__(self, radius=200):
        self.radius = radius

    def calculate_radius(self,X):
        n = len(X)
        sigma = np.std(X)
        r = (4 / (3 * n)) ** (1 / 5) * sigma
        return r

    def fit(self, data):
        self.radius = self.calculate_radius(data)
        print("Radius = ",self.radius)
        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            print("New Iteration")
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwidth.append(featureset)

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids

    def predict(self, data):
            labels = []
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid])
                             for centroid in self.centroids]
                label = distances.index(min(distances))
                labels.append(label)

            self.labels = labels


if __name__ == '__main__':
    print("KMean Shift")
    testPath = "corrected"
    trainPath = "kddcup.data_10_percent_corrected"
    r = DataReader.Reader(trainPath, testPath)
    data, Datalabels, test, testlabels = r.readData()

    X_train, X_labels, y_train, y_test = train_test_split(data, Datalabels, train_size=0.1, random_state=42)

    print(X_train.shape)
    print(X_train)

    ms = MeanShift()
    ms.fit(X_train)
    ms.predict(X_train)
    centroids = ms.centroids
    labels = ms.labels
    print(centroids)
    print(labels)



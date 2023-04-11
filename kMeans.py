import numpy as np
import DataReader
import time
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

def initialize_centroids(data, k):
    np.random.seed(42)
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    clusters = [[] for _ in range(centroids.shape[0])]
    for i, point in enumerate(data):
        centroid_idx = np.argmin(cdist([point], centroids))
        clusters[centroid_idx].append(i)
    return clusters

def update_centroids(data, clusters):
    centroids = np.zeros((len(clusters), data.shape[1]))
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            centroid = np.mean(data[cluster], axis=0)
            centroids[i] = centroid
    return centroids

def predict_clusters(data, centroids):
    clusters = []
    for i, point in enumerate(data):
        centroid_idx = np.argmin(cdist([point], centroids))
        clusters.append(centroid_idx)
    return clusters

def predict_nearest_centroids(data, centroids):
    nearest_centroids = []
    for point in data:
        centroid_idx = np.argmin(cdist([point], centroids))
        nearest_centroids.append(centroid_idx)
    return nearest_centroids

def evaluate(test_data, centroids):
    labels = np.argmin(cdist(test_data, centroids), axis=1)
    return labels

def k_means(data, k, max_iterations=100, tolerance=1e-2):
    centroids = initialize_centroids(data, k)
    prev_centroids = centroids.copy()

    for i in range(max_iterations):
        print(i)
        clusters = assign_clusters(data, centroids)
        centroids = update_centroids(data, clusters)
        centroid_shift = np.sum(np.abs(centroids - prev_centroids))

        if centroid_shift <= tolerance:
            break

        prev_centroids = centroids.copy()

    return centroids, clusters

if __name__ == '__main__':
    testPath = "corrected"
    trainPath = "kddcup.data_10_percent_corrected"
    r = DataReader.Reader(trainPath,testPath)
    trainData, trainLabels, testData, testLabels = r.readData()

    k_values = [7, 15, 23, 31, 45]

    for k in k_values:
        start = time.time()
        centroids, clusters = k_means(trainData, k)
        end = time.time()
        print("     Training Time = ", (end - start) / 60, " mins")

        print(f"KMeans with K={k} finished. Centroids:")
        print(centroids)

        test_nearest_centroids = evaluate(testData,centroids)
        test_accuracy = np.mean(test_nearest_centroids == testLabels) * 100
        print(f"Test accuracy with K={k}: {test_accuracy:.2f}%")
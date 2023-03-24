import random
import numpy as np
import DataReader
import time


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]


def assign_clusters(data, centroids):
    print('assignnnnnnn')
    clusters = [[] for _ in range(centroids.shape[0])]
    for i, point in enumerate(data):
        centroid_idx = np.argmin([euclidean_distance(point, centroid) for centroid in centroids])
        clusters[centroid_idx].append(i)
    return clusters


def update_centroids(data, clusters):
    print('Updaaaaaaate')
    centroids = np.zeros((len(clusters), data.shape[1]))
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            centroid = np.mean(data[cluster], axis=0)
            centroids[i] = centroid
    return centroids


def predict_clusters(data, centroids):
    print('predict')
    clusters = []
    for i, point in enumerate(data):
        centroid_idx = np.argmin([euclidean_distance(point, centroid) for centroid in centroids])
        clusters.append(centroid_idx)
    return clusters


def k_means(data, k, max_iterations=100, tolerance=1e-4):
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

    r = DataReader.Reader()
    trainData , testData = r.readData()


    k_values = [7, 15, 23, 31, 45]

    for k in k_values:
        start = time.time()
        centroids, clusters = k_means(trainData, k)
        end = time.time()
        print("     Training Time = ", (end - start) / 60, " mins")

        print(f"KMeans with K={k} finished. Centroids:")
        print(centroids)
        test_clusters = predict_clusters(testData, centroids)
        print("Test clusters:")
        print(test_clusters)
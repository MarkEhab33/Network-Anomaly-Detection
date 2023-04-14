import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

import ClusterEvaluation
import DataReader
import time
from scipy.spatial.distance import cdist

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



def evaluate(test_data, centroids):
    labels = np.argmin(cdist(test_data, centroids), axis=1)
    return labels

def k_means(data, k, max_iterations=100, tolerance=1e-4):
    centroids = initialize_centroids(data, k)
    prev_centroids = centroids.copy()

    for i in range(max_iterations):

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
    scaler = MinMaxScaler()
    trainData = scaler.fit_transform(trainData)
    testData = scaler.transform(testData)

    k_values = [7,15,23,31,45]

    for k in k_values:
        start = time.time()
        centroids, clusters = k_means(trainData, k)
        end = time.time()
        print(f"KMeans with K={k} finished.")
        print("     Training Time = ", (end - start) / 60, " mins")
        test_nearest_centroids = evaluate(testData,centroids)
       

        # precision = precision_score(test_nearest_centroids, testLabels, average='macro')
        # recall = recall_score(test_nearest_centroids, testLabels, average='macro')
        # f1 = f1_score(test_nearest_centroids, testLabels, average='macro')
        #
        #
        # # Print the results
        #
        # print("Precision: ", precision)
        # print("Recall: ", recall)
        # print("F1 score: ", f1)

        ce = ClusterEvaluation.ClusterEvaluator()


        print("Precision overall value:")
        val1, arr1 = ce.getPrecision(testLabels, test_nearest_centroids)
        print(val1)
        # print("Precision for each cluster:")
        # print(arr1)
        print("--------------------------------")
        print("Recall overall value (ours):")
        val1, arr1 = ce.getRecall(testLabels, test_nearest_centroids)
        print(val1)
        # print("Recall (Fowlkes-Mallows):")
        # print(ce.getOverallRecall(testLabels, test_nearest_centroids))
        # print("Recall for each cluster: ")
        # print(arr1)
        print("--------------------------------")

        print("F1 overall score:")
        val1, arr1 = ce.getF1Score(testLabels, test_nearest_centroids)
        print(val1)
        print("--------------------------------")

        print("Conditional Entropy overall:")
        val1, arr = ce.getConditionalEntropy(testLabels, test_nearest_centroids)
        print(val1)

        # print("Conditional Entropy for each cluster: ")
        # print(arr)

        print("--------------------------------")


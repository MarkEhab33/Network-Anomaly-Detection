import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time
import ClusterEvaluation
import DataReader


class hierarchicalClustering:

    def euclidean_distance(self,x, y):
        distance = cdist([x], [y], metric='euclidean')
        return distance[0][0]

    def distance_matrix(self,X):
        n = X.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = self.euclidean_distance(X[i], X[j])
                D[j, i] = D[i, j]
        return D

    def hierarchical_clustering(self,X, n_clusters):
        D = self.distance_matrix(X)             # calculate the distance matrix
        cluster_assignments = np.arange(X.shape[0])
        cluster_distances = np.zeros(X.shape[0])

        for k in range(X.shape[0] - n_clusters):                    # merge clusters until the desired number of clusters is reached
            i, j = np.unravel_index(np.argmin(D), D.shape)          # closest different pairs
            while i == j:
                D[i, j] = np.inf
                i, j = np.unravel_index(np.argmin(D), D.shape)

            cluster_assignments[cluster_assignments == j] = i
            cluster_distances[i] = D[i, j] / 2

            for l in range(X.shape[0]):         # update the distance matrix
                if l != i and l != j:
                    d_il = D[i, l]
                    d_jl = D[j, l]
                    D[i, l] = np.sqrt((d_il ** 2 + d_jl ** 2 - D[i, j] ** 2) / 2)
                    D[l, i] = D[i, l]
            D[j, :] = np.inf
            D[:, j] = np.inf

        unique_vals, idx = np.unique(cluster_assignments, return_inverse=True)
        new_arr = idx.reshape(cluster_assignments.shape)
        return new_arr, cluster_distances

    def write_labels_to_file(self,labels, filename):
        with open(filename, 'w') as f:
            for label in labels:
                f.write(str(label) + '\n')


if __name__ == '__main__':
    testPath = "corrected"
    trainPath = "kddcup.data.gz\kddcup.data.corrected"
    r = DataReader.Reader(trainPath, testPath)
    data, labels, test, testlabels = r.readData()
    X_train, X_labels, y_train, y_test = train_test_split(data, labels, train_size=0.0015, random_state=42)
    targetLabels = X_train[:, -1]

    clusteringTechnique = hierarchicalClustering()
    evaluator = ClusterEvaluation.ClusterEvaluator()

    clusters = [3,7,11]
    for k in clusters :
        print(" Number of clusters = ",k)
        start = time.time()
        clusterLabels, cluster_distances = clusteringTechnique.hierarchical_clustering(X_train, n_clusters=k)
        end = time.time()
        print("     Training Time = ", (end - start) / 60, " mins")
        print("     Evaluation Results :")
        print("         Precision total result ", evaluator.getPrecision(targetLabels, clusterLabels)[0])
        print("             Precision for each cluster results: ", evaluator.getPrecision(targetLabels, clusterLabels)[1])
        print("         -------------------------------------------------------------------------------------------------")
        print("         Recall total result ", evaluator.getRecall(targetLabels, clusterLabels)[0])
        print("             Recall for each cluster results: ", evaluator.getRecall(targetLabels, clusterLabels)[1])
        print("         -------------------------------------------------------------------------------------------------")
        print("         F1 score total result ",evaluator.getF1Score(targetLabels,clusterLabels)[0])
        print("             F1 score for each cluster results ", evaluator.getF1Score(targetLabels, clusterLabels)[1])
        print("         -------------------------------------------------------------------------------------------------")
        print("         Condition Entropy total result ", evaluator.getConditionalEntropy(targetLabels, clusterLabels)[0])
        print("             Condition Entropy result ", evaluator.getConditionalEntropy(targetLabels, clusterLabels)[1])
        print("         -------------------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------------------------------------")




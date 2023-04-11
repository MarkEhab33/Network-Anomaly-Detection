import numpy as np
import sklearn.metrics.pairwise as pairwise
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

import ClusterEvaluation
import DataReader


class SpectralClustering:
    def fit(self, data, labels):
        self.data = data
        self.labels = labels
        return self

    def getAffinityRBF(self, gamma):
        return pairwise.rbf_kernel(self.data, gamma=gamma)

    def getAffinityCS(self):
        return pairwise.cosine_similarity(self.data)

    def affinity(self, choice='rbf', *args, **kwargs):
        self.affinityStr = choice
        affinity_options = {
            'rbf': self.getAffinityRBF,
            'cos': self.getAffinityCS

        }
        aff_funct = affinity_options[choice]
        self.A = aff_funct(*args, **kwargs)
        return self

    def saveEigs(self, vals, vecs):
        nameVec = 'normcut-' + str(self.affinityStr) + 'vecs' + '.npy'
        nameVal = 'normcut-' + str(self.affinityStr) + 'vals' + '.npy'
        np.save(nameVal, vals)
        np.save(nameVec, vecs)

    def getEigenValsAndVecs(self):
        nameVec = 'normcut-' + str(self.affinityStr) + 'vecs' + '.npy'
        nameVal = 'normcut-' + str(self.affinityStr) + 'vals' + '.npy'
        try:
            vecs = np.load(nameVec)
            vals = np.load(nameVal)
            return vals, vecs
        except FileNotFoundError:
            return self.recalcEigns()

    def recalcEigns(self):
        Delta = self.calculate_delta(self.A)
        Delta_inv = np.linalg.inv(Delta)
        B = np.identity(self.A.shape[0]) - Delta_inv @ self.A
        vl, vc = np.linalg.eig(B)
        eig_vals, eig_vecs = self.sortEign(vl, vc)
        self.saveEigs(eig_vals, eig_vecs)
        return eig_vals, eig_vecs

    def k_way_normalized_cut(self, k):
        eig_vals, eig_vecs = self.getEigenValsAndVecs()
        U = eig_vecs[:, :k]
        self.Y = np.zeros(U.shape)
        epsilon = 1e-7
        for i in range(U.shape[0]):
            row = U[i]
            norm = np.linalg.norm(row) + epsilon
            self.Y[i] = (1 / norm) * row

    def calculate_delta(self, A):
        Delta = np.zeros(A.shape)
        for i in range(A.shape[0]):
            sum = 0
            for j in range(A.shape[1]):
                sum = sum + A[i, j]
            Delta[i, i] = sum
        return Delta

    def sortEign(self, values, vectors, ascending=True):
        direction = 1  # default ascending
        if ascending == False:
            direction = -1
        sortedindicies = np.argsort(values)[::direction]
        sortedValues = values[sortedindicies]
        sortedVectors = vectors[:, sortedindicies]
        return sortedValues, sortedVectors

    def aux_kmeans(self, k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.Y)
        labels = kmeans.labels_

        return labels

    def cluster(self, k):
        self.k_way_normalized_cut(k)
        predict = self.aux_kmeans(k)
        print(predict)
        print(self.labels)
        ce = ClusterEvaluation.ClusterEvaluator()
        print("Precision:")
        val1, arr1 = ce.getPrecision(self.labels,predict)
        print(val1)
        # print(arr1)

        print("Recall (ours)")
        val1, arr1 = ce.getRecall(self.labels,predict)
        print(val1)
        # print(arr1)

        print("Recall (Fowlkes-Mallows):")
        val1 = ce.getOverallRecall(self.labels,predict)
        print(val1)


        print("F1 score:")
        val1, arr1 = ce.getF1Score(self.labels, predict)
        print(val1)

        print("Conditional Entropy:")
        val1 , arr = ce.getConditionalEntropy(self.labels,predict)
        print(val1)

        print("Conditional Entropy for each class: ")
        print(arr)


if __name__ == '__main__':
    testPath = "corrected"
    trainPath = "kddcup.data.gz\kddcup.data.corrected"

    r = DataReader.Reader(trainPath, testPath)
    data, labels, test, testlabels = r.readData()
    X_train, X_labels, y_train, y_test = train_test_split(data, labels, train_size=0.0015, random_state=42)
    spc = SpectralClustering()
    spc.fit(X_train, y_train).affinity('rbf', 0.1).cluster(k=11)

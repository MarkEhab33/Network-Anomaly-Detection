import numpy as np
import sklearn.metrics.pairwise as pairwise
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import DataReader


class SpectralClustering:
    def fit(self, data):
        self.data = data
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
        U = eig_vecs[:, :k - 1]
        self.Y = np.zeros(U.shape)
        for i in range(U.shape[0]):
            row = U[i]
            norm = np.linalg.norm(row)
            self.Y[i] = (1 / norm) * row

    def calculate_delta(self, A):
        Delta = np.zeros(A.shape)
        for i in range(A.shape[0]):
            sum = 0
            for j in range(A.shape[1]):
                sum = sum + A[i, j]
            Delta[i, i] = sum
        return Delta

    def sortEign(self, values, vectors, ascending):
        direction = 1  # default ascending
        if ascending == False:
            direction = -1
        sortedindicies = np.argsort(values)[::direction]
        sortedValues = values[sortedindicies]
        sortedVectors = vectors[:, sortedindicies]
        return sortedValues, sortedVectors

    def dummy_Kmeans(self, k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.Y)
        labels = kmeans.labels_

        return labels

    def cluster(self, k):
        self.k_way_normalized_cut(k)
        print(self.dummy_Kmeans(k))


if __name__ == '__main__':
    testPath = "corrected.gz\corrected"
    trainPath = "kddcup.data.gz\kddcup.data.corrected"

    r = DataReader.Reader(trainPath, testPath)
    data, labels, test, testlabels = r.readData()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.005, random_state=42)
    spc = SpectralClustering()
    spc.fit(X_train).affinity('rbf', 1).cluster()

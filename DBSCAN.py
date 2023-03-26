import DataReader
from sklearn.neighbors import NearestNeighbors # importing the library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DBSCAN_implementation:
    def setParameters(self,eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples


    def expandCluster(self, i, neighbors, cluster_id):
        print("Expand clusteer")
        for neighbor in neighbors:
            if self.labels[neighbor] == -1:
                self.labels[neighbor] = cluster_id
            elif self.labels[neighbor] == 0:
                self.labels[neighbor] = cluster_id
                new_neighbors = self.getNeighbors(neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))
        return

    def getNeighbors(self, i):
        print("Get neighbours of ",i)
        return np.where(np.linalg.norm(self.dataSet - self.dataSet[i], axis=1) < self.eps)[0]
        # condition >> return boolean array where true means that within the eps and false means not in range
        # where >> return true indicies , [0] convert to 1D array

    def dbscan(self, dataSet):
        self.dataSet = dataSet
        self.labels = np.zeros(len(dataSet))
        cluster_id = 1                      # starting with clustering id = 1
        for i in range(len(dataSet)):
            print("new iteration")
            if self.labels[i] == 0:

                neighbors = self.getNeighbors(i)
                if len(neighbors) < self.min_samples:
                    self.labels[i] = -1                 # Noise point
                    continue
                self.labels[i] = cluster_id
                self.expandCluster(i, neighbors, cluster_id)
                cluster_id += 1
        return self.labels

    def write_labels_to_file(self,labels, filename):
        with open(filename, 'w') as f:
            for label in labels:
                f.write(str(label) + '\n')


    def count_elements(self,arr):
        unique_vals, counts = np.unique(arr, return_counts=True)
        return dict(zip(unique_vals, counts))

if __name__ == '__main__':
    testPath = "corrected"
    trainPath = "kddcup.data_10_percent_corrected"
    r = DataReader.Reader(trainPath,testPath)
    data, labels, test, testlabels = r.readData()
   # X_train, X_labels, y_train, y_test = train_test_split(data, labels, train_size=0.0025, random_state=42)
    print(data.shape)
    dbscan = DBSCAN_implementation()
    dbscan.setParameters(500,82)
    labelsClustering = dbscan.dbscan(data)
    dic = dbscan.count_elements(labelsClustering)
    print(dic)


    # print(X_train.shape)
    # dbscan.write_labels_to_file(labelsClustering,"labelsfromdbscan.txt")
    # print("done writing")




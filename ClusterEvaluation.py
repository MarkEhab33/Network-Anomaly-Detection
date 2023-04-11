import numpy as np
from scipy.stats import mode
from itertools import combinations


class ClusterEvaluator:
    def getFN(self, target, predict):
        uniquePred, countClust = np.unique(predict, return_counts=True)
        fn = 0
        for clusterLabel1 in uniquePred[:len(uniquePred) - 1]:
            for clusterLabel2 in uniquePred[clusterLabel1 + 1:]:
                # get the indices of elements in the same cluster
                idx1 = np.where(predict == clusterLabel1)[0]
                idx2 = np.where(predict == clusterLabel2)[0]

                # get the true values of that cluster
                trueClasses1 = target[idx1]
                trueClasses2 = target[idx2]

                uniqueClassInCluster1, countClassInCluster1 = np.unique(trueClasses1, return_counts=True)
                uniqueClassInCluster2, countClassInCluster2 = np.unique(trueClasses2, return_counts=True)

                for i1, classInACluster in enumerate(uniqueClassInCluster1):
                    for i2, classInOtherCluster in enumerate(uniqueClassInCluster2):
                        if (classInACluster == classInOtherCluster):
                            fn = fn + countClassInCluster1[i1] * countClassInCluster2[i2]

        return fn

    def getTP(self, target, predict):
        # get each cluster label and the count of elements in that cluster
        uniquePred, countClust = np.unique(predict, return_counts=True)
        tp = 0
        for clusterLabel in uniquePred:
            # get the indices of elements in the same cluster
            idx = np.where(predict == clusterLabel)[0]

            # get the true values of that cluster
            trueClasses = target[idx]

            uniqueClassInCluster, countClassInCluster = np.unique(trueClasses, return_counts=True)
            for i in range(len(uniqueClassInCluster)):
                # for each class, kCn
                combs = combinations(range(1, countClassInCluster[i] + 1), 2)
                num_combinations = len(list(combs))
                tp = tp + num_combinations

        return tp

    def getOverallRecall(self, target, predict):
        tp = self.getTP(target, predict)
        fn = self.getFN(target, predict)

        return tp / (tp + fn)

    def getPrecision(self, target, predict):
        # get each cluster label and the count of elements in that cluster
        uniquePred, countClust = np.unique(predict, return_counts=True)

        precArr = np.zeros(uniquePred.size)

        for clusterLabel in uniquePred:
            # get the indices of elements in the same cluster
            idx = np.where(predict == clusterLabel)[0]

            # get the true values of that cluster
            trueClasses = target[idx]

            representative = mode(trueClasses, keepdims=False)[0]
            inside = np.count_nonzero(trueClasses == representative)
            all = countClust[clusterLabel]
            precArr[clusterLabel] = inside / all

        totalsize = len(predict)
        precTotal = 0
        for i in range(len(uniquePred)):
            precTotal = precTotal + (countClust[i] / totalsize) * precArr[i]

        return precTotal, precArr

    def getRecall(self, target, predict):
        # separate clusters
        uniquePred, countClust = np.unique(predict, return_counts=True)
        recTotal = 0
        recArr = np.zeros(uniquePred.size)
        totalsize = len(predict)

        for clusterLabel in uniquePred:
            idx = np.where(predict == clusterLabel)[0]
            trueClasses = target[idx]
            # get most repeated class in the cluster
            representative = mode(trueClasses, keepdims=False)[0]
            # how many times is that element in the cluster
            inside = np.count_nonzero(trueClasses == representative)
            # how many of that class is there in all data set
            all = np.count_nonzero(target == representative)
            recArr[clusterLabel] = inside / all
            recTotal = recTotal + (countClust[clusterLabel] / totalsize) * recArr[clusterLabel]

        return recTotal, recArr

    def getF1Score(self, target, predict):
        recTotal, recArr = self.getRecall(target, predict)
        precTotal, precArr = self.getPrecision(target, predict)

        F = np.zeros(len(precArr))
        for i in range(len(precArr)):
            F[i] = (2 * precArr[i] * recArr[i]) / (precArr[i] + recArr[i])
        return np.sum(F) / len(F), F

    def getConditionalEntropy(self, target, predict):
        # get cluster label and their count
        uniquePred, countClust = np.unique(predict, return_counts=True)
        # initialize conditional entropy array
        hTC = np.zeros(len(uniquePred))
        # loop over each cluster
        for clusterLabel in uniquePred:
            # get the true cluster labels corresponding to the elements inside
            idx = np.where(predict == clusterLabel)[0]
            trueClasses = target[idx]
            # get count of each class in the cluster
            classIn, classInCount = np.unique(trueClasses, return_counts=True)
            hTCi = 0
            # loop over each class
            for i in range(len(classIn)):
                # conditional entropy of that class =
                #   count_of_the_class_in_cluster               count_of_the_class_in_cluster
                #   __________________________________ * log   _________________________________
                #   count_of_total_elements_in_cluster          count_of_total_elements_in_cluster
                countR = classInCount[i]
                val = countR / countClust[clusterLabel]
                hTCi += val * np.log10(val)
            # negate the summation
            hTCi = hTCi * (-1)
            hTC[clusterLabel] = hTCi
        totalsize = len(predict)
        hTCTotal = 0
        for i in range(len(uniquePred)):
            hTCTotal = hTCTotal + (countClust[i] / totalsize) * hTC[i]
        return hTCTotal, hTC


if __name__ == '__main__':
    ce = ClusterEvaluator()
    true_labels = np.array([0, 1, 0, 0, 1])
    cluster_labels = np.array([1, 0, 1, 1, 0])

    print("Precision >>>", ce.getPrecision(true_labels, cluster_labels)[0])
    print("Recall old>>>", ce.getRecall(true_labels, cluster_labels)[0])
    print("Recall new>>>", ce.getOverallRecall(true_labels, cluster_labels))
    print("F1 old>>>", ce.getF1Score(true_labels, cluster_labels)[0])
    print("Entropy>>>", ce.getConditionalEntropy(true_labels, cluster_labels)[0])

import numpy as np
from scipy.stats import mode


class ClusterEvaluator:

    def getPrecision(self, target, predict):
        uniquePred, countClust = np.unique(predict, return_counts=True)
        precArr = np.zeros(uniquePred.size)
        for clusterLabel in uniquePred:
            idx = np.where(predict == clusterLabel)[0]
            trueClasses = target[idx]
            representative = mode(trueClasses)[0]
            inside = np.count_nonzero(trueClasses == representative)
            all = countClust[clusterLabel]
            precArr[clusterLabel] = inside / all

        totalsize = len(predict)
        precTotal = 0
        for i in range(len(uniquePred)):
            precTotal = precTotal + (countClust[i] / totalsize) * precArr[i]

        return precTotal, precArr

    def getRecall(self, target, predict):
        uniquePred, countClust = np.unique(predict, return_counts=True)
        recTotal = 0
        recArr = np.zeros(uniquePred.size)
        totalsize = len(predict)

        for clusterLabel in uniquePred:
            idx = np.where(predict == clusterLabel)[0]
            trueClasses = target[idx]
            representative = mode(trueClasses)[0]
            inside = np.count_nonzero(trueClasses == representative)
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
        return np.sum(F) / len(F)

    def getConditionalEntropy(self, target, predict):
        uniquePred, countClust = np.unique(predict, return_counts=True)
        hTC = np.zeros(len(countClust))
        for clusterLabel in uniquePred:
            idx = np.where(predict == clusterLabel)[0]
            trueClasses = target[idx]
            print(trueClasses)
            classIn, classInCount = np.unique(trueClasses, return_counts=True)
            hTCi = 0
            print(classIn)
            print(classInCount)
            for i in range(len(classIn)):
                countR = classInCount[i]
                print(countR)
                val = countR / countClust[clusterLabel]
                print(countClust[clusterLabel])
                hTCi += val * np.log10(val)
            print(hTCi)
            hTCi = hTCi * (-1)
            hTC[clusterLabel] = hTCi
            print(hTC)
            totalsize = len(predict)
            hTCTotal = 0
            for i in range(len(uniquePred)):
                hTCTotal = hTCTotal + (countClust[i] / totalsize) * hTC[i]
                print(hTCTotal)
            return hTCTotal

if __name__ == '__main__':
    ce = ClusterEvaluator()
    true_labels = np.array([0, 1, 0, 0, 1])
    cluster_labels = np.array([1, 0, 1, 1, 0])

    print(">>>", ce.getPrecision(true_labels, cluster_labels)[0])
    print(">>>", ce.getRecall(true_labels, cluster_labels)[0])
    print(">>>", ce.getF1Score(true_labels, cluster_labels))
    print(">>>", ce.getConditionalEntropy(true_labels, cluster_labels))

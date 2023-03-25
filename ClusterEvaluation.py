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

        recArr = np.zeros(uniquePred.size)
        for clusterLabel in uniquePred:
            idx = np.where(predict == clusterLabel)[0]
            trueClasses = target[idx]
            representative = mode(trueClasses)[0]
            inside = np.count_nonzero(trueClasses == representative)
            all = np.count_nonzero(target == representative)
            recArr[clusterLabel] = inside / all

        totalsize = len(predict)
        recTotal = 0
        for i in range(len(uniquePred)):
            recTotal = recTotal + (countClust[i] / totalsize) * recArr[i]

        return recTotal, recArr


    def getF1Score(self, target, predict):
        recTotal, recArr = self.getRecall(target,predict)
        precTotal,precArr = self.getPrecision(target,predict)

        F = np.zeros(len(precArr))
        print("all ",len(precArr) )
        for i in range(len(precArr)):
            print(precArr[i], ",", recArr[i])
            F[i] = (2*precArr[i]*recArr[i])/(precArr[i]+recArr[i])
        print(F)
        return np.sum(F)/len(F)
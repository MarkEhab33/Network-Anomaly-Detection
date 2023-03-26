import pandas as pd


class Reader:
    testPath = ""
    trainPath = ""

    def __init__(self, train, test):
        self.testPath = test
        self.trainPath = train

    def convert_categorical_to_numerical(self, df):
        # loop over all columns in the DataFrame
        for col in df.columns:
            if df[col].dtype == 'object':  # check if column contains categorical data
                df[col] = pd.factorize(df[col])[0]  # convert categorical data to numerical data
        return df

    def readData(self):
        df_Training = pd.read_csv(self.trainPath)
        df_Testing = pd.read_csv(self.testPath)
        df_Training_numerical = self.convert_categorical_to_numerical(df_Training)
        df_Testing_numerical = self.convert_categorical_to_numerical(df_Testing)
        trainingData = df_Training_numerical.to_numpy()
        testingData = df_Testing_numerical.to_numpy()
       # print(trainingData.shape)
        #return training data , training labels , testing data, testing labels
        return trainingData[:, :-1],trainingData[:,-1], testingData[:, :-1],testingData[:,-1]




if __name__ == '__main__':
    testPath = "corrected.gz\corrected"
    trainPath = "kddcup.data.gz\kddcup.data.corrected"

    r = Reader(trainPath, testPath)
    data , dataLabels, test, testLabels = r.readData()
    print(data)
    print(data.shape)

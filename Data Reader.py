import pandas as pd
class Reader:


    def convert_categorical_to_numerical(self, df):
        # loop over all columns in the DataFrame
        for col in df.columns:
            if df[col].dtype == 'object':  # check if column contains categorical data
                df[col] = pd.factorize(df[col])[0]  # convert categorical data to numerical data

        return df

    def readData(self,trainingPath , testingPath):
        df_Training = pd.read_csv(trainingPath)
        df_Testing = pd.read_csv(testingPath)
        df_Training_numerical = self.convert_categorical_to_numerical(df_Training)
        df_Testing_numerical = self.convert_categorical_to_numerical(df_Testing)
        trainingData = df_Training_numerical.to_numpy()
        testingData = df_Testing_numerical.to_numpy()

        return trainingData[1:,1:],testingData[1:,1:]




if __name__ == '__main__':

    testPath = "corrected.gz\corrected"
    trainPath = "kddcup.data.gz\kddcup.data.corrected"

    r = Reader()
    trainData , testData = r.readData(trainPath,testPath)
    #print(trainData.shape)
    # print(trainData.shape)
    # print(testData)
    # print(testData.shape)



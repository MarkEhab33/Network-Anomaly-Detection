import pandas as pd
class Reader:

    def convert_categorical_to_numerical(self, df):
        # loop over all columns in the DataFrame
        for col in df.columns:
            if df[col].dtype == 'object':  # check if column contains categorical data
                df[col] = pd.factorize(df[col])[0]  # convert categorical data to numerical data


        return df

    def readData(self):
        df_Training = pd.read_csv("kddcup.data_10_percent_corrected")
        df_Testing = pd.read_csv("corrected")
        df_Training_numerical = self.convert_categorical_to_numerical(df_Training)
        df_Testing_numerical = self.convert_categorical_to_numerical(df_Testing)
        trainingData = df_Training_numerical.to_numpy()
        testingData = df_Testing_numerical.to_numpy()

        return trainingData[1:,1:],testingData[1:,1:]




if __name__ == '__main__':

    testPath = "corrected"
    trainPath = "kddcup.data.corrected"

    r = Reader()
    trainData , testData = r.readData()
    print(trainData)
    # print(trainData.shape)
    print(testData)
    # print(testData.shape)



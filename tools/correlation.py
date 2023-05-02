import pandas as pd
import scipy
import numpy as np

# data = scipy.io.loadmat('../data/RAP_annotation.mat')
# data_size = 10000


class Correlation:
    def __init__(self, data: object, data_size: int):

        self.features = data['RAP_annotation'][0][0][2]
        self.data_cols = np.zeros(self.features.size, dtype=object)

        for col in range(0, self.features.size):
            self.data_cols[col] = self.features[col][0][0]

        self.data_matrix = np.zeros((data_size, self.data_cols.size), dtype=object)

        for row in range(data_size):
            self.data_matrix[row] = data['RAP_annotation'][0][0][1][row]

        df = pd.DataFrame(self.data_matrix, columns=self.data_cols)
        self.correlation = df.corr().round(3)

    def display(self):
        return self.correlation


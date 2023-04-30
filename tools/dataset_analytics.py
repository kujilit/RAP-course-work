import numpy as np
import pandas as pd


class MakeTable:
    def __init__(self, data, data_size):
        self.bags_cols = data['RAP_annotation'][0][0][2][88:95]

        self.df_cols = np.zeros(self.bags_cols.size + 1, dtype=object)
        self.df_cols[0] = 'name'
        for col in range(1, self.bags_cols.size + 1):
            self.df_cols[col] = self.bags_cols[col - 1][0][0][11:]

        self.data_array = np.zeros((data_size, 8), dtype=object)

        for row in range(data_size):
            new_row = data['RAP_annotation'][0][0][0][row][0][0]
            self.data_array[row][0] = new_row
            self.data_array[row][1:] = data['RAP_annotation'][0][0][1][row][88:95]

        self.df = pd.DataFrame(self.data_array, columns=self.df_cols)

    def display(self):
        return self.df
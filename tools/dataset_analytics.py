import numpy as np
import scipy
import pandas as pd

data = scipy.io.loadmat('../data/RAP_annotation.mat')
bags_cols = data['RAP_annotation'][0][0][2][88:95]

df_cols = np.zeros(bags_cols.size, dtype=str)

for col in range(bags_cols.size):
    df_cols[col] = bags_cols[col][0][0][11:]

data_size = 100
data_array = np.zeros((data_size, 7), dtype=int)

for row in range(data_size):
    data_array[row] = data['RAP_annotation'][0][0][1][row][88:95]

# print(data_array)

df = pd.DataFrame(data_array, columns=df_cols)

print(df)
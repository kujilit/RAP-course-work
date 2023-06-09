import os
from image_to_array import CNN
import numpy as np
import torch
import sys
from sklearn.model_selection import train_test_split
from datetime import datetime


# torch.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=sys.maxsize)
start_time = datetime.now()


torch.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)

path = os.listdir("../dataset")
dataset = CNN(path)

data_len = dataset.__len__()
dataset = dataset.get_conv_images()
#
# x_dataset = []
# y_dataset = []

# for i in range(data_len):
#     x_dataset.append(dataset[i][1])
#     y_dataset.append(dataset[i][2])
#
# x_dataset = np.asarray(x_dataset)
# y_dataset = np.asarray(y_dataset)
#
# x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset)
#
# print(len(x_train), len(y_train), len(x_test), len(y_test))
print(f"Время работы алгоритма: {datetime.now() - start_time}")
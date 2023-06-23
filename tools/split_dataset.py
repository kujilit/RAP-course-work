import os
from image_to_array import CNN

path = os.listdir("../dataset")
dataset = CNN(path)

print(dataset.__len__())
print(dataset.get_conv_images())

from skimage import io
import cv2
import numpy as np
import os

filelist = "../RAP_dataset"


def image2array():
    image_array = []
    for image in os.listdir(filelist):
        img = io.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        image_array.append(img)
    image_array = np.array(image_array)
    image_array = image_array.reshape(image_array.shape[0], 224, 224, 3)
    image_array = image_array.astype('float32')
    image_array /= 255
    return np.array(image_array)


train_data = image2array()
print("Length of training dataset:", train_data.shape)

# print(os.listdir(filelist))

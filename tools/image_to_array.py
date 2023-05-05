from PIL import Image
import numpy as np
import os

path = os.listdir("../dataset")
img_list = np.zeros((len(path)), dtype=object)
print(img_list.size)

img_iter = 0

for element in path:
    img_list[img_iter] = np.asarray(Image.open('../dataset/' + element).convert('RGB'))
    img_iter += 1


# img = np.asarray(Image.open('../RAP_dataset/CAM01-2013-12-23-20131223120147-20131223120735-tarid3-frame493-line1.png').convert('RGB'))

print(img_list)

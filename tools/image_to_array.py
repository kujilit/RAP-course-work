from PIL import Image
import numpy as np
import os

path = os.listdir("../dataset")
img_list = np.zeros((len(path)), dtype=object)
print(img_list.size)

img_iter = 0

for element in path:
    img_list[img_iter] = np.asarray(Image.open('../dataset/' + element).resize((155, 155)).convert('RGB'))
    img_iter += 1

print(img_list)


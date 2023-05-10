from PIL import Image
import numpy as np
import os
import sys


np.set_printoptions(threshold=sys.maxsize)
path = os.listdir("../dataset")
img_list = np.zeros((len(path)), dtype=object)
print(img_list.size)

img_iter = 0

for element in path[:1]:
    img_list[img_iter] = np.pad(np.asarray(Image.open('../dataset/' + element).resize((155, 155)).convert('RGB')), pad_width=1)
    img_iter += 1

print(img_list)

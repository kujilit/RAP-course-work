import traceback

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

path = os.listdir("../dataset")
img_list = np.empty((len(path)), dtype=object)
transform = transforms.ToTensor()

padding = torch.nn.ConstantPad2d(1, 0)

img_iter = 0

kernel = torch.tensor([[
    [0, -1, 0],
    [-1, 1, -1],
    [0, -1, 0]]], dtype=torch.float32)
conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
with torch.no_grad():
    conv.weight.data.copy_(kernel)

for element in path[:1]:
    image = Image.open('../dataset/' + element).resize((255, 255))
    img_list[img_iter] = image
    img_list[img_iter] = padding(transform(img_list[img_iter]))

    try:
        img_list[img_iter] = conv(img_list[img_iter].unsqueeze(0))
    except:
        print("Ошибка применения свертки:", traceback.format_exc())

    img_iter += 1

print(img_list[0].shape)
img_list[0].unsqueeze(0)
output_img = transforms.ToPILImage()(img_list[0][0])
output_img.save("test.jpg")

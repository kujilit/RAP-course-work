import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F

path = os.listdir("../dataset")
img_list = np.empty((len(path)), dtype=object)
transform = transforms.ToTensor()
padding = transforms.Pad(1)
conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, bias=False)

img_iter = 0

kernel = torch.tensor([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=torch.float32)
kernel = kernel.reshape(1, 1, 3, 3)
with torch.no_grad():
    conv.weight = nn.Parameter(kernel)

for element in path:
    image = Image.open('../dataset/' + element).resize((150, 150))
    img_list[img_iter] = transform(image).unsqueeze(0)
    img_list[img_iter] = padding(img_list[img_iter])
    img_list[img_iter] = conv(img_list[img_iter])
    img_list[img_iter].mean().backward()
    #for layer in img_list[img_iter]:
    #    layer = F.conv2d(layer, kernel)
    #    layer[img_iter] = conv(img_list[img_iter])
    #    layer.unsqueeze(0)
    img_iter += 1

print(img_list[0].shape)
img_list[0].unsqueeze(0)
output_img = transforms.ToPILImage()(img_list[0][0])
output_img.save("test.jpg")


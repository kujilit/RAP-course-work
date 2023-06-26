import traceback

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys


# torch.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=sys.maxsize)


class CNN:
    def __init__(self, path):

        global image
        self.path = os.listdir("../dataset")
        self.img_list = np.empty((len(self.path)), dtype=object)
        transform = transforms.ToTensor()

        padding = torch.nn.ConstantPad2d(1, 0)

        self.img_iter = 0

        kernel = torch.tensor([[
            [0, -1, 0],
            [-1, 3, -1],
            [0, -1, 0]]], dtype=torch.float32)
        conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

        with torch.no_grad():
            conv.weight.data.copy_(kernel)

        for element in self.path:
            if element[0] == '.':
                continue
            try:
                image = Image.open('../dataset/' + element).resize((255, 255))
            except:
                print("Не удалось привести изображение к необходимому формату:", traceback.format_exc())

            self.img_list[self.img_iter] = image
            self.img_list[self.img_iter] = padding(transform(self.img_list[self.img_iter]))

            try:
                self.img_list[self.img_iter] = conv(self.img_list[self.img_iter].unsqueeze(0))
            except:
                print("Ошибка применения свертки:", traceback.format_exc())
            self.img_iter += 1

        for i in range(5):
            self.img_list[i].unsqueeze(0)
            output_img = transforms.ToPILImage()(self.img_list[i][0])
            output_img.save(f"test{i}.jpg")

    def get_conv_images(self):
        return self.img_list

    def __len__(self):
        return self.img_iter


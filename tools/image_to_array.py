import traceback
from dataset_analytics import MakeTable
import scipy
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# torch.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=sys.maxsize)


class CNN:
    def __init__(self, path):

        self.path = os.listdir("../dataset")
        data = scipy.io.loadmat('../data/RAP_annotation.mat')
        df = MakeTable(data, len(self.path)).display()

        self.data_table = np.zeros((4354, 3), dtype=object)

        global image
        self.img_list = np.empty((len(self.path)), dtype=object)
        transform = transforms.ToTensor()

        padding = torch.nn.ConstantPad2d(1, 0)

        self.img_iter = 0

        kernel = torch.tensor([[
            [0, -1, 0],
            [-1, 3, -1],
            [0, -1, 0]]], dtype=torch.float32)
        # kernel_2 = torch.tensor([[
        #     [1, 2, 1],
        #     [2, 4, 2],
        #     [1, 2, 1]]], dtype=torch.float32)
        conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        # conv_2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        relu = torch.nn.LeakyReLU(0.2, inplace=True)
        max_pool = torch.nn.MaxPool2d(2, 2)

        with torch.no_grad():
            conv.weight.data.copy_(kernel)
            # conv_2.weight.data.copy_(kernel_2)

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
                # self.img_list[self.img_iter] = (1 / 16) * conv_2(self.img_list[self.img_iter].unsqueeze(0))
                self.img_list[self.img_iter] = relu(self.img_list[self.img_iter])
                self.img_list[self.img_iter] = max_pool(self.img_list[self.img_iter])
                self.data_table[self.img_iter][0] = element
                self.data_table[self.img_iter][1] = self.img_list[self.img_iter]
                signs = []
                line = df.loc[(df['name'] == element)]
                signs.append(line['Backpack'].tolist()[:])
                signs.append(line['ShoulderBag'].tolist()[:])
                signs.append(line['HandBag'].tolist()[:])
                signs.append(line['WaistBag'].tolist()[:])
                signs.append(line['Box'].tolist()[:])
                signs.append(line['PlasticBag'].tolist()[:])
                signs.append(line['PaperBag'].tolist()[:])

                if [1] in signs:
                    self.data_table[self.img_iter][2] = 1
                else:
                    self.data_table[self.img_iter][2] = 0
            except:
                print("Ошибка применения свертки:", traceback.format_exc())
            self.img_iter += 1

        column_names = ['Index', 'Name', 'Backpack', 'ShoulderBag', 'HandBag',
                        'WaistBag', 'Box', 'PlasticBag', 'PaperBag']
        df.to_csv('dataframe.csv', index=False, header=column_names, na_rep='Unknown', )

        for i in range(50, 60):
            self.img_list[i].unsqueeze(0)
            output_img = transforms.ToPILImage()(self.img_list[i][0])
            output_img.save(f"test{60-i}.jpg")

    def get_conv_images(self):
        return self.data_table

    def __len__(self):
        return self.img_iter


path = os.listdir("../dataset")
dataset = CNN(path)

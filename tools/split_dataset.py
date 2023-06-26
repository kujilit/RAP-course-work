import os
from image_to_array import CNN
<<<<<<< HEAD
import torch
import numpy as np
import sys


# torch.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=sys.maxsize)
=======
import numpy as np
import sys
import torch

# torch.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)
>>>>>>> origin/main

path = os.listdir("../dataset")
dataset = CNN(path)

print(dataset.__len__())
print(dataset.get_conv_images())
